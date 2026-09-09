"""Branch registry: the four backbones of the v7 ensemble.

The design point is BitMind's own production model (arXiv:2607.13234, Table 1):
a heterogeneous ensemble whose branches disagree in architecture family,
pretraining objective, and native resolution, fused by FIXED logit averaging.
Their per-branch ablation (Table 23) is the justification: every single branch
has domains below 0.85 AUC (DINOv3's worst is 0.211 on FF++ swaps -- exactly
our fakeclue-fake-ffpp/face-swap failures), while the fused model has zero.

Branches are trained INDEPENDENTLY, one run each, on identical splits and
sampler, then fused at export:

    f(x) = 1/3 * ( 1/2*(g_cnx(x) + g_eva(x)) + g_clip(x) + g_dino(x) )

so a branch can be added, dropped, or retrained without touching the others,
and each gets its own gasbench number. `fusion_weight` below stores the
EFFECTIVE weight (1/6, 1/6, 1/3, 1/3) -- they sum to 1, which matters because
an unnormalised sum silently rescales the baked temperature.

Submission constraints that shaped this file (all verified against gasbench):
  * the harness hands ONE uint8 [B,3,H,W] tensor at the declared resolution;
    per-branch native sizes come from F.interpolate INSIDE the model, so the
    same interpolate must run at train time -- `native_size` is used by
    model.py in both places.
  * `timm` and `transformers` are on the import allowlist; `open_clip` is not.
  * `getattr` is a blocked call in the shipped model.py, so everything the
    template needs (dims, sizes, norm stats) must be exportable as literals --
    which is why this registry holds plain data, not callables.
"""
from __future__ import annotations

from dataclasses import dataclass, field

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


@dataclass(frozen=True)
class BranchSpec:
    name: str                 # config/CLI key and state_dict prefix
    family: str               # "hf_dinov3" | "hf_clip" | "timm" | "dct"
    model_id: str             # HF repo or timm model name
    native_size: int          # F.interpolate target inside the model
    feat_dim: int             # backbone output width feeding the head
    head: str                 # "mac_lite" | "clip_linear" | "bounded_cosine"
    norm_mean: tuple[float, float, float]
    norm_std: tuple[float, float, float]
    fusion_weight: float
    # LoRA target module-name suffixes; empty tuple => head+norms only.
    lora_targets: tuple[str, ...] = ()


# BMF Table 1, with per-branch adaptation choices:
#   dinov3 / clip  -> LoRA on attention projections (proven path)
#   eva            -> LoRA on unfused q/k/v + proj (timm eva02_large builds
#                     with qkv_fused=False -- there is no "qkv" Linear leaf)
#   convnext       -> no attention Linears; train head + LayerNorms only
BRANCHES: dict[str, BranchSpec] = {
    "dinov3": BranchSpec(
        name="dinov3", family="hf_dinov3",
        model_id="facebook/dinov3-vitl16-pretrain-lvd1689m",
        native_size=224, feat_dim=1024, head="mac_lite",
        norm_mean=IMAGENET_MEAN, norm_std=IMAGENET_STD,
        fusion_weight=1 / 3,
        lora_targets=("q_proj", "k_proj", "v_proj", "o_proj")),
    # Resolution-bump experiment variants of dinov3 (EXPECTED_OUTCOMES.md D5).
    # Exact twins except native_size -- DINOv3 uses RoPE (no learned
    # pos-embed), so any /16 multiple is architecturally exact:
    # 336 -> 21x21 = 441 patches (seq 446), 384 -> 24x24 = 576 (581),
    # vs 224 -> 196 (201). fusion_weight is SWAP semantics: a variant ships
    # INSTEAD of dinov3 and inherits its 1/3 slot, so the renormalised 5-way
    # vector stays exactly 2/7,2/7,1/7,1/7,1/7. Never ship dinov3 plus a
    # variant together without re-deriving weights. 336 keeps a mild
    # antialiased 0.875x resample in front (preserves the ensemble's
    # resolution-heterogeneity axis); 384 == data.image_size, so like dct NO
    # resample runs and the branch sees gasbench's non-antialiased
    # INTER_LINEAR aliasing band raw. Separate registry entries on purpose:
    # no tensor shape depends on native_size, so a 224-trained checkpoint
    # would strict-load into a 336/384 model silently -- distinct names get
    # distinct runs/v7/<name>/ dirs and the export.py provenance guard.
    "dinov3_336": BranchSpec(
        name="dinov3_336", family="hf_dinov3",
        model_id="facebook/dinov3-vitl16-pretrain-lvd1689m",
        native_size=336, feat_dim=1024, head="mac_lite",
        norm_mean=IMAGENET_MEAN, norm_std=IMAGENET_STD,
        fusion_weight=1 / 3,
        lora_targets=("q_proj", "k_proj", "v_proj", "o_proj")),
    "dinov3_384": BranchSpec(
        name="dinov3_384", family="hf_dinov3",
        model_id="facebook/dinov3-vitl16-pretrain-lvd1689m",
        native_size=384, feat_dim=1024, head="mac_lite",
        norm_mean=IMAGENET_MEAN, norm_std=IMAGENET_STD,
        fusion_weight=1 / 3,
        lora_targets=("q_proj", "k_proj", "v_proj", "o_proj")),
    "clip": BranchSpec(
        name="clip", family="hf_clip",
        model_id="openai/clip-vit-large-patch14",
        native_size=224, feat_dim=1024, head="clip_linear",
        norm_mean=CLIP_MEAN, norm_std=CLIP_STD,
        fusion_weight=1 / 3,
        lora_targets=("q_proj", "k_proj", "v_proj", "out_proj")),
    "convnext": BranchSpec(
        name="convnext", family="timm",
        model_id="convnext_large_mlp.clip_laion2b_soup_ft_in12k_in1k_384",
        native_size=384, feat_dim=1536, head="bounded_cosine",
        norm_mean=IMAGENET_MEAN, norm_std=IMAGENET_STD,
        fusion_weight=1 / 6,
        # ConvNeXt has no ATTENTION Linears, which is why this was empty --
        # but every block's MLP is built from nn.Linear (timm's Mlp with
        # use_conv=False), so fc1/fc2 are valid LoRA sites. With () the branch
        # trained head+norms only: 0.1M params, and at step 250 it sat at
        # loss 1.25 / acc 0.61 against dinov3's 0.55 / 0.87 -- a capacity
        # ceiling, not slow warmup.
        lora_targets=("fc1", "fc2")),
    "eva": BranchSpec(
        name="eva", family="timm",
        # The 336 variant of this pretrain tag does not exist in timm; 448 is
        # the only eva02_large with mim_m38m_ft_in22k_in1k weights.
        model_id="eva02_large_patch14_448.mim_m38m_ft_in22k_in1k",
        native_size=448, feat_dim=1024, head="bounded_cosine",
        norm_mean=IMAGENET_MEAN, norm_std=IMAGENET_STD,
        fusion_weight=1 / 6,
        lora_targets=("q_proj", "k_proj", "v_proj", "proj")),
    # Frequency-domain forensic branch (ported from video_v1, which follows
    # BMF section 3.3: their specialist fuses "a frequency-domain DCT
    # branch"). Luma -> orthonormal 2-D DCT -> log-magnitude map -> a SMALL
    # convnet trained FROM SCRATCH (no LoRA; ImageNet priors do not transfer
    # to spectra). Generator upsampling leaves periodic energy in exactly
    # this domain, and the branch is nearly blind to content/style -- the
    # complementary signal to the semantic backbones, added 2026-08-10 after
    # dflip3k reals measured 15%: a stylized-real domain the style-prior
    # branches call fake wholesale. ~11M params.
    #
    # native_size == data.image_size (384) ON PURPOSE: the declared input is
    # 384, so no F.interpolate ever runs in front of this branch -- an
    # antialiased downsample would low-pass away the high-frequency band the
    # branch exists to read. Changing data.image_size means revisiting this.
    "dct": BranchSpec(
        name="dct", family="dct",
        model_id="resnet18",           # timm, in_chans=1, pretrained=False
        native_size=384, feat_dim=512, head="bounded_cosine",
        norm_mean=IMAGENET_MEAN, norm_std=IMAGENET_STD,  # unused: DCT maps
        fusion_weight=1 / 6,           # standardize per-sample instead
        lora_targets=()),
}


def resolve(names: list[str]) -> list[BranchSpec]:
    unknown = sorted(set(names) - set(BRANCHES))
    if unknown:
        raise SystemExit(f"unknown branch names {unknown}; "
                         f"registry has {sorted(BRANCHES)}")
    return [BRANCHES[n] for n in names]


def fusion_weights(specs: list[BranchSpec]) -> list[float]:
    """Renormalised weights for the branch subset actually shipped.

    Shipping 2 of 4 branches with raw registry weights would sum to 2/3 and
    silently rescale the baked temperature by the same factor; renormalising
    keeps sum == 1 for any subset, so bake_temperature stays exact.
    """
    w = [s.fusion_weight for s in specs]
    tot = sum(w)
    return [x / tot for x in w]
