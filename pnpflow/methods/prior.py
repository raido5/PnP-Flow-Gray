import torch
import torch.nn.functional as F
from typing import Optional, Tuple


def _ensure_4d(x: torch.Tensor) -> torch.Tensor:
    """
    Convert input to [B, C, H, W].
    Supports:
        [H, W]
        [B, H, W]
        [B, C, H, W]
    """
    if x.dim() == 2:
        x = x.unsqueeze(0).unsqueeze(0)
    elif x.dim() == 3:
        x = x.unsqueeze(1)
    elif x.dim() != 4:
        raise ValueError(f"Unsupported input shape: {x.shape}")
    return x


def _avg_smooth(y: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
    """
    Simple average smoothing for weight estimation.
    y: [B, C, H, W]
    """
    if kernel_size <= 1:
        return y
    pad = kernel_size // 2
    return F.avg_pool2d(y, kernel_size=kernel_size, stride=1, padding=pad)


def _forward_diff_x(x: torch.Tensor) -> torch.Tensor:
    """
    Forward difference along x-direction (width).
    Output shape: [B, C, H, W-1]
    """
    return x[..., 1:] - x[..., :-1]


def _forward_diff_y(x: torch.Tensor) -> torch.Tensor:
    """
    Forward difference along y-direction (height).
    Output shape: [B, C, H-1, W]
    """
    return x[..., 1:, :] - x[..., :-1, :]


def _adjoint_diff_x(p: torch.Tensor, out_shape: torch.Size) -> torch.Tensor:
    """
    Adjoint of forward difference along x-direction.
    p: [B, C, H, W-1]
    return: [B, C, H, W]
    """
    out = torch.zeros(out_shape, dtype=p.dtype, device=p.device)
    out[..., :-1] -= p
    out[..., 1:] += p
    return out


def _adjoint_diff_y(p: torch.Tensor, out_shape: torch.Size) -> torch.Tensor:
    """
    Adjoint of forward difference along y-direction.
    p: [B, C, H-1, W]
    return: [B, C, H, W]
    """
    out = torch.zeros(out_shape, dtype=p.dtype, device=p.device)
    out[..., :-1, :] -= p
    out[..., 1:, :] += p
    return out


def build_adaptive_weights(
    y: torch.Tensor,
    tau: Optional[float] = None,
    smooth_kernel_size: int = 5,
    weight_mode: str = "exp",
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build adaptive edge-aware weights W_x, W_y from observed image y.

    Args:
        y: observed image, shape [H,W], [B,H,W], or [B,C,H,W]
        tau: edge sensitivity. If None, use 2 * median gradient magnitude.
        smooth_kernel_size: smoothing size before computing gradients
        weight_mode: "exp", "charbonnier", or "inverse"
        eps: small constant for stability

    Returns:
        wx: weights for horizontal differences, shape [B,C,H,W-1]
        wy: weights for vertical differences, shape [B,C,H-1,W]
    """
    y = _ensure_4d(y)
    y_s = _avg_smooth(y, kernel_size=smooth_kernel_size)

    gy_x = _forward_diff_x(y_s).abs()
    gy_y = _forward_diff_y(y_s).abs()

    if tau is None:
        # Robust automatic choice
        all_g = torch.cat([gy_x.reshape(-1), gy_y.reshape(-1)], dim=0)
        tau = 2.0 * torch.median(all_g).item()
        tau = max(tau, 1e-6)

    tau2 = tau * tau

    if weight_mode == "exp":
        wx = torch.exp(-(gy_x ** 2) / (tau2 + eps))
        wy = torch.exp(-(gy_y ** 2) / (tau2 + eps))
    elif weight_mode == "charbonnier":
        wx = 1.0 / (1.0 + (gy_x ** 2) / (tau2 + eps))
        wy = 1.0 / (1.0 + (gy_y ** 2) / (tau2 + eps))
    elif weight_mode == "inverse":
        wx = 1.0 / torch.sqrt(gy_x ** 2 + tau2 + eps)
        wy = 1.0 / torch.sqrt(gy_y ** 2 + tau2 + eps)
    else:
        raise ValueError(f"Unknown weight_mode: {weight_mode}")

    return wx, wy


def apply_adaptive_Q(
    x: torch.Tensor,
    wx: torch.Tensor,
    wy: torch.Tensor,
) -> torch.Tensor:
    """
    Compute Qx = D_x^T W_x D_x x + D_y^T W_y D_y x
    without explicitly forming the large matrix Q.

    Args:
        x: image, shape [H,W], [B,H,W], or [B,C,H,W]
        wx: horizontal weights, shape [B,C,H,W-1]
        wy: vertical weights, shape [B,C,H-1,W]

    Returns:
        qx: Qx, same shape as x after normalization to 4D
    """
    x4 = _ensure_4d(x)

    gx = _forward_diff_x(x4)   # [B,C,H,W-1]
    gy = _forward_diff_y(x4)   # [B,C,H-1,W]

    wxgx = wx * gx
    wygy = wy * gy

    qx = _adjoint_diff_x(wxgx, x4.shape) + _adjoint_diff_y(wygy, x4.shape)
    return qx


def adaptive_quadratic_prior(
    x: torch.Tensor,
    y: Optional[torch.Tensor] = None,
    wx: Optional[torch.Tensor] = None,
    wy: Optional[torch.Tensor] = None,
    lam: float = 1.0,
    tau: Optional[float] = None,
    smooth_kernel_size: int = 5,
    weight_mode: str = "exp",
    eps: float = 1e-8,
    return_energy: bool = True,
):
    """
    Full wrapper:
        1) If wx, wy are not provided, build them from observed image y.
        2) Compute Qx.
        3) Return lambda * Qx, and optionally the prior energy:
              R(x) = lam/2 * x^T Q x

    Args:
        x: current estimate, [H,W], [B,H,W], or [B,C,H,W]
        y: observed image used to build adaptive weights
        wx, wy: optional precomputed weights
        lam: regularization strength lambda
        tau: edge sensitivity
        smooth_kernel_size: smoothing size before computing weights
        weight_mode: "exp", "charbonnier", or "inverse"
        eps: small constant
        return_energy: whether to return R(x)

    Returns:
        prior_grad: lam * Qx, same 4D shape as x
        energy: scalar tensor if return_energy=True
        wx, wy: weights
    """
    x4 = _ensure_4d(x)

    if wx is None or wy is None:
        if y is None:
            raise ValueError("Either provide (wx, wy) or provide observed image y.")
        wx, wy = build_adaptive_weights(
            y=y,
            tau=tau,
            smooth_kernel_size=smooth_kernel_size,
            weight_mode=weight_mode,
            eps=eps,
        )

    qx = apply_adaptive_Q(x4, wx, wy)
    prior_grad = lam * qx

    if not return_energy:
        return prior_grad, wx, wy

    # R(x) = lam/2 * ( ||sqrt(Wx) Dx x||^2 + ||sqrt(Wy) Dy x||^2 )
    gx = _forward_diff_x(x4)
    gy = _forward_diff_y(x4)
    energy = 0.5 * lam * ((wx * gx * gx).sum() + (wy * gy * gy).sum())

    return prior_grad, energy, wx, wy