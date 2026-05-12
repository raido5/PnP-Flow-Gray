import torch
import torch.nn.functional as F
import numpy as np
import os
from time import perf_counter
import pnpflow.image_generation.models.utils as mutils
import pnpflow.utils as utils
from pnpflow.methods.prior import build_adaptive_weights, adaptive_quadratic_prior


def _get_arg(args, name, default):
    return getattr(args, name, default)


def _diff_kernels(channels, device, dtype):
    # filtres différence horizontale et verticale
    fh = torch.tensor([[0.,0.,0.],[0.,1.,-1.],[0.,0.,0.]], device=device, dtype=dtype).view(1,1,3,3)
    fv = torch.tensor([[0.,0.,0.],[0.,1.,0.],[0.,-1.,0.]], device=device, dtype=dtype).view(1,1,3,3)
    return fh.repeat(channels,1,1,1), fv.repeat(channels,1,1,1)


def gsm_l1_prior_grad(x, eps=1e-3, use_log_domain=True, normalize=False, return_energy=False):
    """L1/Laplace GSM sparse prior gradient. lambda(theta) = 1/(|Dz| + eps)"""
    import torch.nn.functional as F
    B, C, H, W = x.shape
    x_safe = x.clamp_min(eps)
    z = torch.log(x_safe) if use_log_domain else x
    fh, fv = _diff_kernels(C, z.device, z.dtype)
    gh = F.conv2d(z, fh, padding=1, groups=C)
    gv = F.conv2d(z, fv, padding=1, groups=C)
    # L1 weights : lambda = 1 / (|Dz| + eps)
    lam_h = (1.0 / (gh.abs() + eps)).detach()
    lam_v = (1.0 / (gv.abs() + eps)).detach()
    grad_z = 2.0 * (
        F.conv_transpose2d(lam_h * gh, fh, padding=1, groups=C)
        + F.conv_transpose2d(lam_v * gv, fv, padding=1, groups=C)
    )
    if normalize:
        grad_z = grad_z / float(B * C * H * W)
    grad_x = grad_z / x_safe if use_log_domain else grad_z
    if return_energy:
        energy = (lam_h * gh.pow(2) + lam_v * gv.pow(2)).sum() / float(B)
        return grad_x, energy
    return grad_x


class PNP_FLOW(object):

    def __init__(self, model, device, args):
        self.device = device
        self.args = args
        self.model = model.to(device)
        self.method = args.method

    def model_forward(self, x, t):
        if self.args.model == "ot":
            return self.model(x, t)

        elif self.args.model == "rectified":
            model_fn = mutils.get_model_fn(self.model, train=False)
            t_ = t[:, None, None, None]
            v = model_fn(x.type(torch.float), t * 999)
            return v

    def learning_rate_strat(self, lr, t):
        t = t.view(-1, 1, 1, 1)
        gamma_styles = {
            '1_minus_t': lambda lr, t: lr * (1 - t),
            'sqrt_1_minus_t': lambda lr, t: lr * torch.sqrt(1 - t),
            'constant': lambda lr, t: lr,
            'alpha_1_minus_t': lambda lr, t: lr * (1 - t)**self.args.alpha,
        }
        return gamma_styles.get(self.args.gamma_style, lambda lr, t: lr)(lr, t)

    def grad_datafit(self, x, y, H, H_adj):
        if self.args.noise_type == 'gaussian':
            return H_adj(H(x) - y) / (self.args.sigma_noise**2) #Gradient pour débruiter
        elif self.args.noise_type == 'laplace':
            return H_adj(2*torch.heaviside(H(x)-y, torch.zeros_like(H(x)))-1)/self.args.sigma_noise
        elif self.args.noise_type == 'gamma':
            gradient = self.args.sigma_noise * (x - y)
            use_gsm = _get_arg(self.args, 'use_gsm_prior', True)
            if use_gsm:
                gsm_grad, gsm_energy = gsm_l1_prior_grad(
                    x=x,
                    eps=_get_arg(self.args, 'gsm_prior_eps', 1e-3),
                    use_log_domain=_get_arg(self.args, 'gsm_prior_log_domain', True),
                    normalize=_get_arg(self.args, 'gsm_prior_normalize', False),
                    return_energy=True,
                )
                gradient = gradient + _get_arg(self.args, 'gsm_prior_beta', 1e-3) * gsm_grad
                self.args.last_gsm_energy = float(gsm_energy.detach().cpu())
            return gradient
        else:
            raise ValueError('Noise type not supported')

    def interpolation_step(self, x, t):
        return t * x + torch.randn_like(x) * (1 - t)

    def denoiser(self, x, t):
        v = self.model_forward(x, t)
        return x + (1 - t.view(-1, 1, 1, 1)) * v

    def solve_ip(self, test_loader, degradation, sigma_noise, H_funcs=None):
        H = degradation.H
        H_adj = degradation.H_adj
        self.args.sigma_noise = sigma_noise
        num_samples = self.args.num_samples
        steps, delta = self.args.steps_pnp, 1 / self.args.steps_pnp
        if self.args.noise_type == 'gaussian':
            self.args.lr_pnp = sigma_noise**2 * self.args.lr_pnp
            lr = self.args.lr_pnp
        elif self.args.noise_type == 'laplace':
            self.args.lr_pnp = sigma_noise * self.args.lr_pnp
            lr = self.args.lr_pnp
        elif self.args.noise_type == 'gamma':
            lr = _get_arg(self.args, 'lr_gamma_pnp', 0.5)
        else:
            raise ValueError('Noise type not supported')

        loader = iter(test_loader)
        for batch in range(self.args.max_batch):

            (clean_img, labels) = next(loader)
            self.args.batch = batch
            print(clean_img.shape)

            if self.args.noise_type == 'gaussian':
                noisy_img = H(clean_img.clone().to(self.device))
                torch.manual_seed(batch)
                noisy_img += torch.randn_like(noisy_img) * sigma_noise #NOISE ADDED
            elif self.args.noise_type == 'laplace':
                noisy_img = H(clean_img.clone().to(self.device))
                noise = torch.distributions.laplace.Laplace(
                    torch.zeros_like(noisy_img), sigma_noise * torch.ones_like(noisy_img)).sample().to(self.device)
                noisy_img += noise
            elif self.args.noise_type == 'gamma':
                # H(x)=x
                noisy_img = H(clean_img.clone().to(self.device))
                # Here sigma_noise is the look number (L)
                # when set alpha and beta equal to L
                # the mean and variance of the gamma noise will be 1 and 1/L respectively,
                # which is a common setting for simulating speckle noise in SAR images
                alpha = beta = sigma_noise
                noise = torch.distributions.Gamma(concentration=torch.tensor(alpha),
                            rate=torch.tensor(beta)).sample(sample_shape=noisy_img.shape).to(self.device)
                noisy_img = noisy_img * noise
                # If we clamp the noisy image to [0, 1], it may cause issues with the gamma noise model,
                # as the noise can naturally produce values greater than 1.
                # Clamping would distort the noise distribution
                # Therefore, we should avoid clamping the noisy image in this case.
                # noisy_img = torch.clamp(noisy_img, 0, 1)
            else:
                raise ValueError('Noise type not supported')

            noisy_img, clean_img = noisy_img.to(
                self.device), clean_img.to('cpu')

            # intialize the image with the adjoint operator
            x = H_adj(torch.ones_like(noisy_img)).to(self.device)

            if self.args.compute_time:
                torch.cuda.synchronize()
                time_per_batch = 0

            if self.args.compute_memory:
                torch.cuda.reset_max_memory_allocated(self.device)

            with torch.no_grad():
                for count, iteration in enumerate(range(int(steps))):
                    if self.args.compute_time:
                        time_counter_1 = perf_counter()

                    t1 = torch.ones(
                        len(x), device=self.device) * delta * iteration
                    lr_t = self.learning_rate_strat(lr, t1)

                    z = x - lr_t * \
                        self.grad_datafit(x, noisy_img, H, H_adj) #ICI

                    x_new = torch.zeros_like(x)
                    for _ in range(num_samples):
                        z_tilde = self.interpolation_step(
                            z, t1.view(-1, 1, 1, 1))
                        x_new += self.denoiser(z_tilde, t1)

                    x_new /= num_samples
                    x = x_new

                    if self.args.noise_type == 'gamma' and _get_arg(self.args, 'gamma_clamp_after_denoise', True):
                        x = x.clamp_min(_get_arg(self.args, 'gamma_x_min', 1e-6))

                    if self.args.compute_time:
                        torch.cuda.synchronize()
                        time_counter_2 = perf_counter()
                        time_per_batch += time_counter_2 - time_counter_1

                    if self.args.save_results:
                        if iteration % 50 == 0 or self.should_save_image(iteration, steps):

                            restored_img = x.detach().clone()
                            # utils.save_images(
                            #     clean_img, noisy_img, restored_img, self.args, H_adj, iter=iteration)
                            utils.compute_psnr(clean_img, noisy_img,
                                               restored_img, self.args, H_adj, iter=iteration)
                            utils.compute_ssim(
                                clean_img, noisy_img, restored_img, self.args, H_adj, iter=iteration)
                            utils.compute_lpips(clean_img, noisy_img,
                                                restored_img, self.args, H_adj, iter=iteration)

            if self.args.compute_memory:
                dict_memory = {}
                dict_memory["batch"] = batch
                dict_memory["max_allocated"] = torch.cuda.max_memory_allocated(
                    self.device)
                utils.save_memory_use(dict_memory, self.args)

            if self.args.compute_time:
                dict_time = {}
                dict_time["batch"] = batch
                dict_time["time_per_batch"] = time_per_batch
                utils.save_time_use(dict_time, self.args)

            if self.args.save_results:
                restored_img = x.detach().clone()
                utils.save_images(clean_img, noisy_img, restored_img,
                                  self.args, H_adj, iter='final')
                utils.compute_psnr(clean_img, noisy_img,
                                   restored_img, self.args, H_adj, iter=iteration)
                utils.compute_ssim(
                    clean_img, noisy_img, restored_img, self.args, H_adj, iter=iteration)
                utils.compute_lpips(clean_img, noisy_img,
                                    restored_img, self.args, H_adj, iter=iteration)

        if self.args.save_results:
            utils.compute_average_psnr(self.args)
            utils.compute_average_ssim(self.args)
            utils.compute_average_lpips(self.args)
        if self.args.compute_memory:
            utils.compute_average_memory(self.args)
        if self.args.compute_time:
            utils.compute_average_time(self.args)

    def should_save_image(self, iteration, steps):
        return iteration % (steps // 10) == 0

    def run_method(self, data_loaders, degradation, sigma_noise, H_funcs=None):

        # Construct the save path for results
        folder = utils.get_save_path_ip(self.args.dict_cfg_method)
        self.args.save_path_ip = os.path.join(self.args.save_path, folder)

        # Create the directory if it doesn't exist
        os.makedirs(self.args.save_path_ip, exist_ok=True)

        # Solve the inverse problem
        self.solve_ip(
            data_loaders[self.args.eval_split], degradation, sigma_noise, H_funcs)
