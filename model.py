"""v7 branch model: one backbone + one head, trained alone, fused at export.

Loss surface is deliberately small -- CE on both views, KL consistency between
them, and a ramped Brier term. The v5 spectral teacher / MIL / supcon /
deep-supervision stack is gone: none of it was ever individually ablated, BMF's
production branches carry none of it, and a lean loss makes per-branch results
attributable.

Heads (BMF section 3.1, widened to the 3-class image taxonomy):
  mac_lite       -- CLS ++ mean(patch tokens) -> MLP(2d -> 512 -> 3). Branch C.
  clip_linear    -- dropout(0.3) on pooled CLS -> Linear(d, 3). Branch B.
  bounded_cosine -- L2-normalised features & weights, cosine * s=30, squashed
                    through 15*tanh(x/15), fixed per-class bias.
                    Branches A1/A2. The tanh CAPS per-branch confidence, which
                    is a Brier control: no single branch can drive the fused
                    logit to an overconfident wrong answer.

The forward contract is the deployed one: uint8 [B,3,H,W] in, [B,3] logits
out, ordered [real, synthetic, semisynthetic] (gasbench image class indices
0/1/2). Interpolation to the branch's native size and its own normalisation
happen INSIDE forward, so train and eval preprocessing are byte-identical by
construction.

Every head's semisynthetic logit is initialised ~3 nats below the synthetic
logit, so an UNTRAINED (or type_weight=0) head already says
p(semi|fake) ~ sigmoid(-3) ~ 0.05 -- the benchmark's semisynthetic share of
the fake half -- instead of an indecisive 0.5 that would bleed multiclass
Brier across the ~48%-share synthetic class.

The binary axis is always read through the margin helpers below, NEVER as
z[:,1] - z[:,0]: with 3 logits the real-vs-fake margin is
logsumexp(z1, z2) - z0, and sigmoid of THAT equals softmax(z)[1:] mass
exactly. Calibration, spread, export, and eval must all share this one
definition (fuse per-branch margins, never logsumexp of fused logits).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from branches import BranchSpec


# ---------------------------------------------------------------------------
# LoRA (hand-rolled; `peft` is not on the submission import allowlist)
# ---------------------------------------------------------------------------

class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: int, dropout: float):
        super().__init__()
        self.base = base
        self.scaling = alpha / r
        self.drop = nn.Dropout(dropout)
        self.A = nn.Linear(base.in_features, r, bias=False)
        self.B = nn.Linear(r, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.B.weight)

    def forward(self, x):
        return self.base(x) + self.scaling * self.B(self.A(self.drop(x)))

    @torch.no_grad()
    def merged(self) -> nn.Linear:
        out = nn.Linear(self.base.in_features, self.base.out_features,
                        bias=self.base.bias is not None)
        delta = (self.B.weight.float() @ self.A.weight.float()) * self.scaling
        out.weight.copy_((self.base.weight.float() + delta)
                         .to(self.base.weight.dtype))
        if self.base.bias is not None:
            out.bias.copy_(self.base.bias)
        return out


def inject_lora(root: nn.Module, targets: tuple[str, ...],
                r: int, alpha: int, dropout: float) -> dict[str, int]:
    """Wrap matching nn.Linear leaves; returns per-target match counts.

    Per-target (not total) so a partially wrong target list -- e.g. a timm
    variant whose attention uses q_proj/k_proj/v_proj where branches.py says
    qkv -- fails loudly instead of training silently degraded.
    """
    per_target = {t: 0 for t in targets}
    for name, mod in list(root.named_modules()):
        leaf = name.rsplit(".", 1)[-1]
        if leaf in targets and isinstance(mod, nn.Linear):
            parent = root.get_submodule(name.rsplit(".", 1)[0]) if "." in name else root
            setattr(parent, leaf, LoRALinear(mod, r, alpha, dropout))
            per_target[leaf] += 1
    return per_target


def merge_lora(root: nn.Module) -> int:
    n = 0
    for name, mod in list(root.named_modules()):
        if isinstance(mod, LoRALinear):
            parent = root.get_submodule(name.rsplit(".", 1)[0]) if "." in name else root
            setattr(parent, name.rsplit(".", 1)[-1], mod.merged())
            n += 1
    return n


# ---------------------------------------------------------------------------
# 3-class margin helpers -- the single source of truth for every consumer
# (train metrics, calibration, export forward, measure/eval tooling).
# ---------------------------------------------------------------------------

def binary_margin(z: torch.Tensor) -> torch.Tensor:
    """[B,K>=2] logits -> [B] real-vs-not-real margin.

    sigmoid(binary_margin(z)) == softmax(z)[:, 1:].sum(-1) exactly, for any K.
    This is the quantity all 1-D calibration (T, hinge, Platt) operates on.
    """
    return torch.logsumexp(z[:, 1:], dim=1) - z[:, 0]


def type_margin(z: torch.Tensor) -> torch.Tensor:
    """[B,3] logits -> [B] semi-vs-synthetic margin (conditional on fake).

    sigmoid(type_margin(z)) == p(semisynthetic | not-real).
    """
    return z[:, 2] - z[:, 1]


def collapse_binary(z: torch.Tensor) -> torch.Tensor:
    """[B,K>=2] logits -> [B,2] collapsed [real, not-real] logits.

    softmax(collapse_binary(z)) == [p_real, 1-p_real] of softmax(z), so
    2-class CE on the collapse (with label smoothing) has exactly the
    binary-training semantics the 2-logit pipeline had.
    """
    return torch.stack([z[:, 0], torch.logsumexp(z[:, 1:], dim=1)], dim=1)


def _prior_shift_bias_(fc: nn.Linear) -> None:
    """Init a 3-class head bias to [0, 0, -3]: untrained p(semi|fake) ~ 0.05."""
    with torch.no_grad():
        fc.bias.zero_()
        fc.bias[2] = -3.0


# ---------------------------------------------------------------------------
# Heads
# ---------------------------------------------------------------------------

class MACLiteHead(nn.Module):
    """CLS ++ mean(patch tokens) -> MLP. Token-sequence backbones only."""

    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(2 * dim)
        self.mlp = nn.Sequential(nn.Linear(2 * dim, 512), nn.GELU(),
                                 nn.Linear(512, 3))
        _prior_shift_bias_(self.mlp[2])

    def forward(self, cls: torch.Tensor, patches: torch.Tensor):
        return self.mlp(self.norm(torch.cat([cls, patches.mean(dim=1)], -1)))


class CLIPLinearHead(nn.Module):
    def __init__(self, dim: int, p: float = 0.3):
        super().__init__()
        self.drop = nn.Dropout(p)
        self.fc = nn.Linear(dim, 3)
        _prior_shift_bias_(self.fc)

    def forward(self, pooled: torch.Tensor):
        return self.fc(self.drop(pooled))


class BoundedCosineHead(nn.Module):
    """L2-normalised cosine classifier with tanh-bounded logits.

    z = 15 * tanh(s * cos(theta) / 15) + [0, +0.5, -2.5], s = 30. The +0.5
    keeps the original real-vs-fake prior on BOTH fake logits; the -2.5 puts
    the semi logit ~3 nats under synthetic (prior-shift init; bias applies
    after the tanh, so per-class range shifts to at most ~[-17.5, +15.5] --
    still bounded). softmax confidence stays capped and, more to the point,
    a WRONG branch contributes a bounded penalty to the fused average instead
    of an unbounded one.
    """

    def __init__(self, dim: int, s: float = 30.0, bound: float = 15.0):
        super().__init__()
        self.s, self.bound = s, bound
        self.weight = nn.Parameter(torch.empty(3, dim))
        nn.init.xavier_normal_(self.weight)
        self.register_buffer("fake_bias", torch.tensor([0.0, 0.5, -2.5]))

    def forward(self, feat: torch.Tensor):
        cos = F.linear(F.normalize(feat, dim=-1), F.normalize(self.weight, dim=-1))
        return self.bound * torch.tanh(self.s * cos / self.bound) + self.fake_bias


# ---------------------------------------------------------------------------
# Frequency-domain front end (dct branch)
# ---------------------------------------------------------------------------


def dct_basis(n: int) -> torch.Tensor:
    """Orthonormal DCT-II basis matrix D [n, n]: X_dct = D @ X @ D.T.

    Pure tensor construction (also used verbatim in the submission template,
    where scipy is avoided and `getattr` is blocked). Orthonormal so the
    transform is an isometry -- verify.py asserts D @ D.T == I."""
    k = torch.arange(n, dtype=torch.float64).view(-1, 1)
    i = torch.arange(n, dtype=torch.float64).view(1, -1)
    d = torch.cos(math.pi / n * k * (i + 0.5)) * math.sqrt(2.0 / n)
    d[0] *= math.sqrt(0.5)
    return d.to(torch.float32)


def images_to_dct(x01: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """[N,3,s,s] float in [0,1] -> [N,1,s,s] standardized log-DCT magnitude.

    Luma (ITU-R 601) -> 2-D DCT -> log1p|.| -> per-sample standardize. The
    standardization replaces dataset normalisation constants: spectra vary
    orders of magnitude with content, and per-sample zero-mean/unit-var is
    what keeps the scratch convnet's input in range for every corpus."""
    luma = (0.299 * x01[:, 0] + 0.587 * x01[:, 1] + 0.114 * x01[:, 2])
    spec = basis @ luma @ basis.transpose(0, 1)
    m = torch.log1p(spec.abs()).unsqueeze(1)
    mu = m.mean(dim=(2, 3), keepdim=True)
    sd = m.std(dim=(2, 3), keepdim=True).clamp_min(1e-5)
    return (m - mu) / sd


# ---------------------------------------------------------------------------
# Branch model
# ---------------------------------------------------------------------------

class BranchModel(nn.Module):
    """One backbone + one head behind the deployed forward contract."""

    def __init__(self, spec: BranchSpec, *, lora_r: int = 32,
                 lora_alpha: int = 64, lora_dropout: float = 0.05,
                 gradient_checkpointing: bool = False,
                 pretrained: bool = True):
        super().__init__()
        self.spec = spec
        self.family = spec.family

        if spec.family == "hf_dinov3":
            from transformers import AutoModel
            self.backbone = (AutoModel.from_pretrained(spec.model_id)
                             if pretrained else None)
        elif spec.family == "hf_clip":
            from transformers import CLIPVisionModel
            self.backbone = (CLIPVisionModel.from_pretrained(spec.model_id)
                             if pretrained else None)
        elif spec.family == "timm":
            import timm
            self.backbone = timm.create_model(
                spec.model_id, pretrained=pretrained, num_classes=0)
        elif spec.family == "dct":
            import timm
            # Always from scratch: ImageNet priors do not transfer to
            # log-DCT spectra, and the net is small enough to train fully.
            # zero_init_last=False: timm's default zero-inits each block's
            # last BN gamma, which makes every inner conv/bn's gradient
            # EXACTLY zero at step 0 -- train.py's dead-gradient check would
            # abort a run that is actually fine. Init-only; the template
            # rebuild loads trained weights and needs no kwarg.
            self.backbone = timm.create_model(
                spec.model_id, pretrained=False, num_classes=0, in_chans=1,
                zero_init_last=False)
        else:
            raise ValueError(f"unknown family {spec.family!r}")
        if self.backbone is None:
            raise ValueError("pretrained=False is only for offline rebuild "
                             "via the exported template, not training")

        dim = spec.feat_dim
        if spec.head == "mac_lite":
            self.head = MACLiteHead(dim)
        elif spec.head == "clip_linear":
            self.head = CLIPLinearHead(dim)
        elif spec.head == "bounded_cosine":
            self.head = BoundedCosineHead(dim)
        else:
            raise ValueError(f"unknown head {spec.head!r}")

        if spec.family == "dct":
            pass  # scratch-trained: every backbone parameter stays trainable
        else:
            for p in self.backbone.parameters():
                p.requires_grad = False
            if spec.lora_targets:
                hits = inject_lora(self.backbone, spec.lora_targets,
                                   lora_r, lora_alpha, lora_dropout)
                print(f"[lora] {spec.name}: " +
                      ", ".join(f"{t} x{c}" for t, c in hits.items()))
                missed = [t for t, c in hits.items() if c == 0]
                if missed:
                    raise RuntimeError(
                        f"LoRA target(s) {missed} matched no nn.Linear in "
                        f"{spec.model_id} (hits: {hits}); the branch would "
                        f"train silently degraded. Fix branches.py.")
            # norms stay trainable regardless -- cheap and consistently helpful
            for name, p in self.backbone.named_parameters():
                if "norm" in name.lower() or "layer_scale" in name.lower() \
                        or name.endswith(".gamma"):
                    p.requires_grad = True

        if gradient_checkpointing:
            if hasattr(self.backbone, "gradient_checkpointing_enable"):
                # Non-reentrant + input-grad hook. With the embeddings frozen,
                # reentrant checkpointing sees no grad-requiring inputs at the
                # first checkpointed block and silently drops the LoRA
                # gradients inside it -- the branch would train head-only.
                self.backbone.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False})
                if hasattr(self.backbone, "enable_input_require_grads"):
                    self.backbone.enable_input_require_grads()
            elif hasattr(self.backbone, "set_grad_checkpointing"):
                self.backbone.set_grad_checkpointing(True)

        self.register_buffer("pixel_mean",
                             torch.tensor(spec.norm_mean).view(1, 3, 1, 1))
        self.register_buffer("pixel_std",
                             torch.tensor(spec.norm_std).view(1, 3, 1, 1))
        if spec.family == "dct":
            self.register_buffer("dct_d", dct_basis(spec.native_size))

    # -- preprocessing: identical at train and eval by construction ---------
    def _prep(self, x_uint8: torch.Tensor) -> torch.Tensor:
        p = x_uint8.to(torch.float32) / 255.0
        s = self.spec.native_size
        if p.shape[-1] != s or p.shape[-2] != s:
            # antialias matters: eval feeds a larger declared resolution down
            # to native, and a plain bilinear decimation would alias exactly
            # the high-frequency band the detector keys on.
            p = F.interpolate(p, size=(s, s), mode="bilinear",
                              antialias=True, align_corners=False)
        if self.family == "dct":
            p = images_to_dct(p, self.dct_d.to(torch.float32))
        else:
            p = (p - self.pixel_mean.to(torch.float32)) \
                / self.pixel_std.to(torch.float32)
        return p.to(next(self.head.parameters()).dtype)

    def _features(self, p: torch.Tensor):
        if self.family == "hf_dinov3":
            out = self.backbone(pixel_values=p)
            h = out.last_hidden_state          # [B, 1+R+N, D], post-norm
            n_reg = 4
            return {"cls": h[:, 0], "patches": h[:, 1 + n_reg:]}
        if self.family == "hf_clip":
            out = self.backbone(pixel_values=p)
            return {"pooled": out.pooler_output}
        # timm with num_classes=0 -> pooled feature vector
        return {"pooled": self.backbone(p)}

    def forward(self, x_uint8: torch.Tensor) -> torch.Tensor:
        f = self._features(self._prep(x_uint8))
        if isinstance(self.head, MACLiteHead):
            return self.head(f["cls"], f["patches"])
        return self.head(f["pooled"])

    # -- training utilities -------------------------------------------------
    def param_groups(self, lr: float) -> list[dict]:
        params = [p for p in self.parameters() if p.requires_grad]
        return [{"params": params, "lr": lr}]

    def merge_and_strip(self) -> int:
        """Fold LoRA into base weights; the module becomes export-shaped."""
        return merge_lora(self.backbone)
