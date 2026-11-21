"""MLP and FNO mapping the measured field to the 8 Bernstein coefficients (about 65k parameters each).

Input channels on the 13-frame x 20-bin grid (12): c - 1 and v for each of the 3 currents, frame mask,
normalised t, normalised x, iota / 3 for each current.  c - 1 and v are divided by their training RMS.
  MLP  1576 -> 40 -> 40 -> 8 (GELU): the c - 1 and v maps (6 x 260), the per-frame mask (13) and iota / 3 (3);
       x and t are implicit in the flattened position.                                   65,048 parameters
  FNO  1x1 lift to width 11 -> 4 Fourier layers (5 x 6 modes, zero-padded to 16 x 24) -> 1x1 projection ->
       mean + max pooling over the valid frames -> 44 -> 128 -> 8.                        65,807 parameters
Output: 8 logits z; theta = sigmoid(z).
"""
from __future__ import annotations

import torch
import torch.nn as nn

N_CHANNELS = 12
MASK = 6


def features(yv, iotas, scales=None):
    """yv (B, 6, 13, 20) = [c (3), v (3)], NaN on invalid frames -> (B, 12, 13, 20)."""
    B, _, T, X = yv.shape
    m = torch.isfinite(yv[:, 0, :, 0])[:, None, :, None].to(yv.dtype)
    tt = torch.linspace(0, 1, T, dtype=yv.dtype)[None, None, :, None].expand(B, 1, T, X)
    xx = torch.linspace(0, 1, X, dtype=yv.dtype)[None, None, None, :].expand(B, 1, T, X)
    f = torch.cat([torch.nan_to_num(yv[:, :3] - 1.0) * m, torch.nan_to_num(yv[:, 3:]) * m, m.expand(B, 1, T, X), tt, xx,
                   (iotas / 3.0)[:, :, None, None].expand(B, 3, T, X)], 1)
    return f if scales is None else f / scales[None, :, None, None]


def input_scales(yv, iotas):
    """RMS of each channel over the valid frames of the training set; 1 for mask, t, x and iota."""
    with torch.no_grad():
        f = features(yv, iotas)
        m = f[:, MASK:MASK + 1].expand_as(f)
        s = torch.sqrt((f ** 2 * m).sum((0, 2, 3)) / m.sum((0, 2, 3)))
        s[MASK:] = 1.0
        return s.clamp(min=1e-8)


class MLP(nn.Module):
    def __init__(self, scales, hidden=40):
        super().__init__()
        self.register_buffer("scales", torch.as_tensor(scales))
        self.net = nn.Sequential(nn.Linear(6 * 13 * 20 + 13 + 3, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Linear(hidden, 8))

    def forward(self, yv, iotas):
        f = features(yv, iotas, self.scales)
        return self.net(torch.cat([f[:, :6].flatten(1), f[:, MASK, :, 0], iotas / 3.0], 1))


class SpectralConv2d(nn.Module):
    def __init__(self, cin, cout, m1, m2):
        super().__init__()
        self.m1, self.m2 = m1, m2
        sc = 1.0 / (cin * cout)
        self.w1 = nn.Parameter(sc * torch.randn(cin, cout, m1, m2, dtype=torch.cfloat))
        self.w2 = nn.Parameter(sc * torch.randn(cin, cout, m1, m2, dtype=torch.cfloat))

    def forward(self, x):
        B, C, Hh, W = x.shape
        xf = torch.fft.rfft2(x)
        out = torch.zeros(B, self.w1.shape[1], Hh, W // 2 + 1, dtype=torch.cfloat)
        out[:, :, :self.m1, :self.m2] = torch.einsum("bixy,ioxy->boxy", xf[:, :, :self.m1, :self.m2], self.w1)
        out[:, :, -self.m1:, :self.m2] = torch.einsum("bixy,ioxy->boxy", xf[:, :, -self.m1:, :self.m2], self.w2)
        return torch.fft.irfft2(out, s=(Hh, W))


def masked_pool(h, mask):
    """Mean and max over the valid (t, x) points: h (B, C, T, X), mask (B, 1, T, X) -> (B, 2C)."""
    mean = (h * mask).sum((2, 3)) / mask.sum((2, 3)).clamp(min=1.0)
    mx = torch.where(mask > 0, h, torch.full_like(h, -1e4)).amax((2, 3))
    return torch.cat([mean, mx], 1)


class FNO(nn.Module):
    def __init__(self, scales, width=11, modes=(5, 6), layers=4, pad=(3, 4)):
        super().__init__()
        self.pad = pad
        self.register_buffer("scales", torch.as_tensor(scales))
        self.lift = nn.Conv2d(N_CHANNELS, width, 1)
        self.spec = nn.ModuleList([SpectralConv2d(width, width, *modes) for _ in range(layers)])
        self.pw = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(layers)])
        self.proj = nn.Conv2d(width, 2 * width, 1)
        self.head = nn.Sequential(nn.Linear(4 * width, 128), nn.GELU(), nn.Linear(128, 8))

    def forward(self, yv, iotas):
        x = features(yv, iotas, self.scales)
        mask = x[:, MASK:MASK + 1]
        x = nn.functional.pad(self.lift(x), (0, self.pad[1], 0, self.pad[0]))
        for k, (sp, pw) in enumerate(zip(self.spec, self.pw)):
            x = sp(x) + pw(x)
            if k < len(self.spec) - 1:
                x = nn.functional.gelu(x)
        x = nn.functional.gelu(self.proj(x[:, :, :yv.shape[2], :yv.shape[3]]))
        return self.head(masked_pool(x, mask))


def build(arch, scales, seed):
    torch.manual_seed(seed)
    return MLP(scales) if arch == "mlp" else FNO(scales)


def n_params(model):
    return sum(p.numel() * (2 if p.is_complex() else 1) for p in model.parameters())


@torch.no_grad()
def predict(model, S, batch=512):
    """theta (n, 8) for every material of a split."""
    return torch.cat([torch.sigmoid(model(S["yv"][s:s + batch], S["it"][s:s + batch])) for s in range(0, len(S["yv"]), batch)])
