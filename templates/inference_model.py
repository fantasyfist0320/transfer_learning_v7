"""SN34 image discriminator. Output: [B, 3] logits ordered
[real, synthetic, semisynthetic] (log-probabilities; softmax recovers them).

Heterogeneous ensemble, factorized 3-class output. Input is a single uint8
[B, 3, H, W] batch; each branch resamples to its own native resolution and
applies its own normalisation internally. Per-branch real-vs-fake margins are
fused and temperature-scaled (with a disagreement-aware hinge) into p_fake;
a conditional syn-vs-semi posterior q splits the fake mass. All calibration
constants are buffers in the weights file; no runtime parameter is passed and
none should be added.

Every architecture constant below is a substituted literal. Nothing is looked
up from a config object at load time except the one Hugging Face config.json
shipped for the self-supervised branch, and the only attribute read from it is
plain dotted access.
"""
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModel, CLIPVisionConfig, CLIPVisionModel
__TIMM_IMPORT__

BRANCH_DEFS: List[dict] = __BRANCH_DEFS__
FUSION_W: List[float] = __FUSION_W__
IMAGE_SIZE: int = __IMAGE_SIZE__
CLIP_CFG: dict = __CLIP_CFG__
DINO_NUM_REGISTERS: int = 4


class MACLiteHead(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(2 * dim)
        self.mlp = nn.Sequential(nn.Linear(2 * dim, 512), nn.GELU(),
                                 nn.Linear(512, 3))

    def forward(self, cls: torch.Tensor, patches: torch.Tensor):
        return self.mlp(self.norm(torch.cat([cls, patches.mean(dim=1)], -1)))


class CLIPLinearHead(nn.Module):
    def __init__(self, dim: int, p: float = 0.3):
        super().__init__()
        self.drop = nn.Dropout(p)
        self.fc = nn.Linear(dim, 3)

    def forward(self, pooled: torch.Tensor):
        return self.fc(self.drop(pooled))


class BoundedCosineHead(nn.Module):
    def __init__(self, dim: int, s: float = 30.0, bound: float = 15.0):
        super().__init__()
        self.s, self.bound = s, bound
        self.weight = nn.Parameter(torch.empty(3, dim))
        self.register_buffer("fake_bias", torch.tensor([0.0, 0.5, -2.5]))

    def forward(self, feat: torch.Tensor):
        cos = F.linear(F.normalize(feat, dim=-1),
                       F.normalize(self.weight, dim=-1))
        return self.bound * torch.tanh(self.s * cos / self.bound) + self.fake_bias


def dct_basis(n: int) -> torch.Tensor:
    """Orthonormal DCT-II basis D [n, n]: X_dct = D @ X @ D.T (pure torch)."""
    k = torch.arange(n, dtype=torch.float64).view(-1, 1)
    i = torch.arange(n, dtype=torch.float64).view(1, -1)
    d = torch.cos(3.141592653589793 / n * k * (i + 0.5)) * ((2.0 / n) ** 0.5)
    d[0] = d[0] * (0.5 ** 0.5)
    return d.to(torch.float32)


class BranchNet(nn.Module):
    """One backbone + head. State keys mirror the training BranchModel."""

    def __init__(self, d: dict, model_dir: str):
        super().__init__()
        self.family = d["family"]
        self.native = d["native"]
        self.head_kind = d["head"]
        if self.family == "hf_dinov3":
            config = AutoConfig.from_pretrained(model_dir,
                                                local_files_only=True)
            self.backbone = AutoModel.from_config(config)
        elif self.family == "hf_clip":
            self.backbone = CLIPVisionModel(CLIPVisionConfig(**CLIP_CFG))
        elif self.family == "dct":
            self.backbone = timm.create_model(d["timm_id"], pretrained=False,
                                              num_classes=0, in_chans=1)
            self.register_buffer("dct_d", dct_basis(d["native"]))
        else:
            self.backbone = timm.create_model(d["timm_id"], pretrained=False,
                                              num_classes=0)
        if self.head_kind == "mac_lite":
            self.head = MACLiteHead(d["dim"])
        elif self.head_kind == "clip_linear":
            self.head = CLIPLinearHead(d["dim"])
        else:
            self.head = BoundedCosineHead(d["dim"])
        self.register_buffer("pixel_mean",
                             torch.tensor(d["mean"]).view(1, 3, 1, 1))
        self.register_buffer("pixel_std",
                             torch.tensor(d["std"]).view(1, 3, 1, 1))

    def forward(self, x_uint8: torch.Tensor) -> torch.Tensor:
        p = x_uint8.to(torch.float32) / 255.0
        if p.shape[-1] != self.native or p.shape[-2] != self.native:
            p = F.interpolate(p, size=(self.native, self.native),
                              mode="bilinear", antialias=True,
                              align_corners=False)
        if self.family == "dct":
            luma = (0.299 * p[:, 0] + 0.587 * p[:, 1] + 0.114 * p[:, 2])
            basis = self.dct_d.to(torch.float32)
            spec = basis @ luma @ basis.transpose(0, 1)
            m = torch.log1p(spec.abs()).unsqueeze(1)
            mu = m.mean(dim=(2, 3), keepdim=True)
            sd = m.std(dim=(2, 3), keepdim=True).clamp_min(1e-5)
            p = (m - mu) / sd
        else:
            p = (p - self.pixel_mean.to(torch.float32)) \
                / self.pixel_std.to(torch.float32)
        p = p.to(next(self.head.parameters()).dtype)
        if self.family == "hf_dinov3":
            h = self.backbone(pixel_values=p).last_hidden_state
            return self.head(h[:, 0], h[:, 1 + DINO_NUM_REGISTERS:])
        if self.family == "hf_clip":
            return self.head(self.backbone(pixel_values=p).pooler_output)
        return self.head(self.backbone(p))


class Ensemble(nn.Module):
    def __init__(self, model_dir: str):
        super().__init__()
        self.branches = nn.ModuleList(
            [BranchNet(d, model_dir) for d in BRANCH_DEFS])
        self.register_buffer("w", torch.tensor(FUSION_W,
                                               dtype=torch.float32))
        self.register_buffer("temperature", torch.tensor(1.0))
        self.register_buffer("temp_slope", torch.tensor(0.0))
        self.register_buffer("spread_pivot", torch.tensor(0.0))
        self.register_buffer("type_temperature", torch.tensor(1.0))
        self.register_buffer("type_bias", torch.tensor(-3.0))
        self.register_buffer("q_min", torch.tensor(1e-3))
        self.register_buffer("q_max", torch.tensor(1.0))
        self.register_buffer("p_min", torch.tensor(1e-6))
        self.register_buffer("p_max", torch.tensor(1.0 - 1e-6))

    def forward(self, x_uint8: torch.Tensor) -> torch.Tensor:
        zs = [m(x_uint8).to(torch.float32) for m in self.branches]
        d = torch.stack([torch.logsumexp(zi[:, 1:], dim=1) - zi[:, 0]
                         for zi in zs], dim=1)
        w = self.w.to(torch.float32).view(1, -1)
        dbar = (w * d).sum(dim=1)
        spread = ((w * (d - dbar.unsqueeze(1)) ** 2).sum(dim=1)).sqrt()
        excess = torch.clamp(spread - self.spread_pivot.to(torch.float32),
                             min=0.0)
        t = self.temperature.to(torch.float32) * (
            1.0 + self.temp_slope.to(torch.float32) * excess)
        p_fake = torch.clamp(torch.sigmoid(dbar / t),
                             self.p_min.to(torch.float32),
                             self.p_max.to(torch.float32))
        d2 = torch.stack([zi[:, 2] - zi[:, 1] for zi in zs], dim=1)
        d2bar = (w * d2).sum(dim=1)
        q = torch.clamp(
            torch.sigmoid(d2bar / self.type_temperature.to(torch.float32)
                          + self.type_bias.to(torch.float32)),
            self.q_min.to(torch.float32), self.q_max.to(torch.float32))
        p = torch.stack([1.0 - p_fake, p_fake * (1.0 - q), p_fake * q],
                        dim=1)
        return torch.log(p)


def load_model(weights_path: str, num_classes: int = 3, **kwargs) -> nn.Module:
    model_dir = Path(weights_path).parent
    model = Ensemble(str(model_dir))
    model.load_state_dict(load_file(weights_path), strict=True)
    model.train(False)
    return model
