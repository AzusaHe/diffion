"""Inference from binned dimensionless concentrations, observation times and currents."""
import hashlib
import io
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


FRAMES = {20: 13, 40: 26, 80: 52, 160: 104, 290: 151}


class SpectralBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = nn.Parameter(torch.empty(11, 11, 5, 6, dtype=torch.cfloat))
        self.w2 = nn.Parameter(torch.empty_like(self.w1))

    def forward(self, x):
        spectrum = torch.fft.rfft2(x)
        low = torch.einsum("biuv,iouv->bouv", spectrum[:, :, :5, :6], self.w1)
        high = torch.einsum("biuv,iouv->bouv", spectrum[:, :, -5:, :6], self.w2)
        middle = low.new_zeros(x.shape[0], 11, x.shape[-2] - 10, 6)
        packed = F.pad(torch.cat((low, middle, high), dim=2), (0, spectrum.shape[-1] - 6))
        return torch.fft.irfft2(packed, s=x.shape[-2:])


class FNO(nn.Module):
    def __init__(self, currents):
        super().__init__()
        self.currents = currents
        self.lift = nn.Conv2d(5 * currents + 1, 11, 1)
        self.spec = nn.ModuleList(SpectralBlock() for _ in range(4))
        self.pw = nn.ModuleList(nn.Conv2d(11, 11, 1) for _ in range(4))
        self.proj = nn.Conv2d(11, 22, 1)
        self.head = nn.Sequential(nn.Linear(44, 128), nn.GELU(), nn.Linear(128, 8))

    def forward(self, features):
        k = self.currents
        mask = features[:, 2*k:3*k].amax(1, keepdim=True)
        x = F.pad(self.lift(features), (0, 4, 0, 3))
        for i, (spectral, pointwise) in enumerate(zip(self.spec, self.pw)):
            x = spectral(x) + pointwise(x)
            if i < 3:
                x = F.gelu(x)
        nt, nx = features.shape[-2:]
        x = F.gelu(self.proj(x[:, :, :nt, :nx]))
        mean = (x * mask).sum((2, 3)) / mask.sum((2, 3)).clamp_min(1)
        maximum = torch.where(mask > 0, x, torch.full_like(x, -1e4)).amax((2, 3))
        return self.head(torch.cat((mean, maximum), 1))


class MLP(nn.Module):
    def __init__(self, currents, width):
        super().__init__()
        self.currents = currents
        self.width = width
        if width > 20:
            self.conv = nn.Conv2d(4, 3, 3, stride=(2, 3), padding=1)
        self.net = nn.Sequential(nn.Linear(currents * 547, 40), nn.GELU(),
                                 nn.Linear(40, 40), nn.GELU(), nn.Linear(40, 8))

    def forward(self, features):
        k = self.currents
        current = features[:, 4*k:5*k, 0, 0]
        if self.width == 20:
            return self.net(torch.cat((features[:, :2*k].flatten(1),
                                       features[:, 2*k:4*k, :, 0].flatten(1), current), 1))
        b, _, nt, nx = features.shape
        fields = features[:, :4*k].reshape(b, 4, k, nt, nx).transpose(1, 2)
        fields = fields.reshape(b*k, 4, nt, nx)[:, [0, 1, 3, 2]]
        valid = fields[:, 3:4, :, 0]
        for _ in range({40: 0, 80: 1, 160: 2, 290: 2}[self.width]):
            first = F.gelu(F.conv2d(fields, self.conv.weight, self.conv.bias, stride=2, padding=1))
            valid = F.max_pool1d(valid, 3, stride=2, padding=1)
            mask = valid.unsqueeze(-1).expand(-1, 1, -1, first.shape[-1])
            fields = torch.cat((first * mask, mask), 1)
        if self.width == 290:
            pooled = F.adaptive_avg_pool2d(fields, (26, 40))
            coverage = pooled[:, 3:4]
            mask = (coverage > 0).to(fields.dtype)
            means = pooled[:, :3] / torch.where(coverage > 0, coverage, 1.0)
            fields = torch.cat((means * mask, mask), 1)
            valid = mask[:, :, :, 0]
        encoded = F.gelu(self.conv(fields))
        valid = F.max_pool1d(valid, 3, stride=2, padding=1)
        encoded = (encoded * valid.unsqueeze(-1)).reshape(b, k, -1)
        return self.net(torch.cat((encoded, current.unsqueeze(-1)), dim=2).flatten(1))


def volume_coefficient(c):
    """Known dimensionless partial-volume term used to reconstruct velocity."""
    u = (c / 2).clamp(0, 1)
    k = c.new_tensor((0.084741, 0.218525, 0.431779, 0.698133))
    basis = torch.stack(((1-u)**3, 3*u*(1-u)**2, 3*u**2*(1-u), u**3), -1)
    quadratic = torch.stack(((1-u)**2, 2*u*(1-u), u**2), -1)
    rho = 1000 + 1500 * (basis * k).sum(-1)
    slope = 1500 * (quadratic * (3 * (k[1:] - k[:-1]))).sum(-1) * 0.5
    return (0.28709 * 3000 - slope) / (rho - c * slope)


def features_from_observations(c, tau, iota, valid, input_rms):
    """Assemble the exact training channel order using saved per-current scales."""
    c = torch.where(valid[..., None], c, 1.0)
    tau = torch.where(valid, tau, 0.0)
    pairs = valid[:, :, 1:] & valid[:, :, :-1]
    delta = torch.where(pairs, tau.diff(dim=2), 1.0)
    if not (delta > 0).all():
        raise ValueError("Adjacent valid frame times must increase.")
    ct = c.diff(dim=2) / delta[..., None] * pairs[..., None]
    zero = torch.zeros_like(c[:, :, :1])
    rate = torch.cat((ct, zero), 2) + torch.cat((zero, ct), 2)
    divisor = F.pad(pairs.float(), (0, 1)) + F.pad(pairs.float(), (1, 0))
    rate = rate / divisor[..., None].clamp_min(1)
    nx = c.shape[-1]
    faces = torch.cat((c[..., :1], (c[..., 1:] + c[..., :-1]) / 2, c[..., -1:]), -1)
    flux = iota[:, :, None, None] - F.pad(rate.cumsum(-1) / nx, (1, 0))
    volume = volume_coefficient(c)
    velocities = [torch.zeros_like(flux[..., 0])]
    q = flux[..., 0]
    for j in range(nx):
        velocities.append((velocities[-1] - volume[..., j] * (flux[..., j+1] - q)) /
                          (1 - volume[..., j] * faces[..., j+1]))
        q = flux[..., j+1] - faces[..., j+1] * velocities[-1]
    v = torch.stack([(a+b)/2 for a, b in zip(velocities[:-1], velocities[1:])], -1)
    mask = valid[..., None].expand_as(c)
    x = torch.linspace(0, 1, nx, device=c.device, dtype=c.dtype)
    return torch.cat(((c-1) / input_rms[0, :, None, None] * mask,
                      v / input_rms[1, :, None, None] * mask, mask,
                      tau[..., None].expand_as(c) / (36000/33640),
                      iota[:, :, None, None].expand_as(c) / 5,
                      x[None, None, None, :].expand(len(c), 1, c.shape[2], nx)), 1)


class Predictor(nn.Module):
    """Map binned fields [batch, currents, time, space] to eight coefficients."""
    def __init__(self, architecture, width, currents, input_rms):
        super().__init__()
        if width not in FRAMES or currents not in (1, 3):
            raise ValueError(f"Unknown grid or current count: {width}, {currents}")
        if architecture not in ("fno", "mlp" if width == 20 else "conv_mlp"):
            raise ValueError(f"Unknown architecture {architecture!r} for {width} bins")
        self.width, self.currents = width, currents
        self.network = FNO(currents) if architecture == "fno" else MLP(currents, width)
        rms = torch.as_tensor(input_rms, dtype=torch.float32)
        if rms.shape != (2, currents) or not torch.isfinite(rms).all() or not torch.all(rms > 0):
            raise ValueError("input_rms must contain positive concentration and velocity scales per current.")
        self.register_buffer("input_rms", rms)

    def forward(self, concentration, tau, iota, mask=None):
        """Inputs use the benchmark's dimensionless units. Always returns [batch, 8]."""
        device = self.input_rms.device
        c, time, current = (torch.as_tensor(a, dtype=torch.float32, device=device)
                            for a in (concentration, tau, iota))
        valid = torch.isfinite(c).all(-1) if mask is None else torch.as_tensor(mask, dtype=torch.bool, device=device)
        if c.ndim == 3:
            c, time, current, valid = (a.unsqueeze(0) for a in (c, time, current, valid))
        if c.ndim != 4 or tuple(c.shape[1:]) != (self.currents, FRAMES[self.width], self.width):
            raise ValueError(f"Expected [batch, {self.currents}, {FRAMES[self.width]}, {self.width}] concentrations.")
        if time.shape != c.shape[:-1] or valid.shape != time.shape or current.shape != c.shape[:2]:
            raise ValueError("Time, mask or current shape does not match concentration.")
        if not valid.any(-1).all() or not torch.isfinite(c[valid]).all() or not torch.isfinite(time[valid]).all() or not torch.isfinite(current).all():
            raise ValueError("Every selected current requires finite observations and times.")
        features = features_from_observations(c, time, current, valid, self.input_rms)
        return self.network(features).sigmoid()


def load_checkpoint(checkpoint_directory):
    """Load the manifest and its complete PyTorch state_dict on CPU in evaluation mode."""
    path = Path(checkpoint_directory)
    manifest = path / "manifest.json" if path.is_dir() else path
    meta = json.loads(manifest.read_text(encoding="utf-8"))
    width, frames = meta["grid_space_time"]
    if FRAMES.get(width) != frames:
        raise ValueError(f"Unsupported space/time grid {width} × {frames}")
    inputs = meta["inputs"]
    model = Predictor(meta["architecture"], width, meta["currents"],
                      [inputs["concentration_deviation_rms"], inputs["velocity_rms"]])
    weights = manifest.parent / meta["checkpoint"]["file"]
    stored = weights.read_bytes()
    if hashlib.sha256(stored).hexdigest() != meta["checkpoint"]["sha256"]:
        raise ValueError(f"Checkpoint SHA-256 differs from its manifest: {weights}")
    state = torch.load(io.BytesIO(stored), map_location="cpu", weights_only=True)
    model.network.load_state_dict(state, strict=True)
    return model.eval()
