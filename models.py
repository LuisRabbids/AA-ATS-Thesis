"""
models.py
Anatomically-Biased Adaptive Masked Autoencoder (manuscript Sec. 3.4).

Contents
--------
  PatchEmbed              Sec. 3.4.1 part 1   (Eq. 3.1)
  TransformerBlock        standard pre-norm ViT block
  TokenSampler            Sec. 3.4.1 part 2 + Sec. 3.4.3  (Eq. 3.2-3.3, 3.10-3.16)
  MaskedAutoencoder       Sec. 3.4.1 / 3.4.4  (Eq. 3.4, 3.5, 3.17-3.20)

Four masking strategies are selectable via `mask_mode`:
  'random'      standard MAE, uniform random masking                       [baseline]
  'adaptive'    AdaMAE, learned sampler over patch embeddings only         [baseline]
  'anatomical'  proposed AA-ATS, sampler over embeddings fused with A_i    [ours]
  'hard_anat'   rule-based hard masking driven directly by A_i, no learning
                -- a *proxy* for fixed-prior methods, NOT a reimplementation of
                   API-MAE or AMAP (those require atlas registration).

Note on ViT-Base: depth 12, dim 768, 12 heads (Sec. 3.4.1 part 3). The decoder is
deliberately lightweight (dim 384, depth 4) per the MAE design.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------------------
# Positional embeddings
# --------------------------------------------------------------------------------------
def sincos_pos_embed(dim, grid):
    """Fixed 2D sine-cosine positional embeddings -> [grid*grid, dim]."""
    def _1d(d, pos):
        omega = 1.0 / 10000 ** (torch.arange(d // 2, dtype=torch.float32) / (d / 2.0))
        out = pos.reshape(-1)[:, None] * omega[None, :]
        return torch.cat([torch.sin(out), torch.cos(out)], dim=1)

    g = torch.arange(grid, dtype=torch.float32)
    gy, gx = torch.meshgrid(g, g, indexing="ij")
    return torch.cat([_1d(dim // 2, gx), _1d(dim // 2, gy)], dim=1)  # [grid^2, dim]


# --------------------------------------------------------------------------------------
# Building blocks
# --------------------------------------------------------------------------------------
class PatchEmbed(nn.Module):
    """Eq. 3.1: image -> non-overlapping patches -> linear projection X_P in R^{N x D}."""

    def __init__(self, img_size=224, patch=16, in_ch=4, dim=768):
        super().__init__()
        self.grid = img_size // patch
        self.n_patches = self.grid**2
        self.patch = patch
        self.in_ch = in_ch
        self.proj = nn.Conv2d(in_ch, dim, kernel_size=patch, stride=patch)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)  # [B, N, D]


class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4.0):
        super().__init__()
        self.n1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.n2 = nn.LayerNorm(dim)
        h = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, h), nn.GELU(), nn.Linear(h, dim))

    def forward(self, x):
        y = self.n1(x)
        x = x + self.attn(y, y, y, need_weights=False)[0]
        return x + self.mlp(self.n2(x))


# --------------------------------------------------------------------------------------
# Anatomically-Aware Adaptive Token Sampler (Sec. 3.4.3)
# --------------------------------------------------------------------------------------
class TokenSampler(nn.Module):
    """
    Produces a categorical distribution P over the N tokens.

    fusion (Sec. 3.4.3 part 1), used when mode == 'anatomical':
      'direct'    Eq. 3.10:  x~_i = MLP([x_i || A_i])
      'learnable' Eq. 3.11:  A~_i = gamma * A_i + beta        (element-wise)
                  Eq. 3.12:  x~_i = MLP([x_i || A~_i])

    Scoring (Eq. 3.2-3.3, 3.13-3.14):
      Z = MHA(X~);  z_i = f_theta(x~_i);  p_i = softmax(z)_i
    """

    def __init__(self, dim=768, n_anat=4, heads=8, fusion="learnable", use_anat=True):
        super().__init__()
        self.use_anat = use_anat
        self.fusion = fusion

        if use_anat:
            if fusion == "learnable":
                # gamma, beta: learnable per-feature scale and shift (Eq. 3.11)
                self.gamma = nn.Parameter(torch.ones(n_anat))
                self.beta = nn.Parameter(torch.zeros(n_anat))
            self.fuse = nn.Sequential(
                nn.Linear(dim + n_anat, dim), nn.GELU(), nn.Linear(dim, dim)
            )

        self.norm = nn.LayerNorm(dim)
        self.mha = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.score = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim // 4),
                                   nn.GELU(), nn.Linear(dim // 4, 1))

    def forward(self, x, anat=None, temperature=1.0):
        """x: [B, N, D], anat: [B, N, F] -> probs [B, N]"""
        if self.use_anat and anat is not None:
            a = anat * self.gamma + self.beta if self.fusion == "learnable" else anat
            x = self.fuse(torch.cat([x, a], dim=-1))  # Eq. 3.10 / 3.12

        y = self.norm(x)
        z = x + self.mha(y, y, y, need_weights=False)[0]      # Eq. 3.2
        logits = self.score(z).squeeze(-1) / temperature       # Eq. 3.13
        return F.softmax(logits, dim=-1)                       # Eq. 3.3 / 3.14


# --------------------------------------------------------------------------------------
# Masked Autoencoder
# --------------------------------------------------------------------------------------
class MaskedAutoencoder(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch=16,
        in_ch=4,
        dim=768,
        depth=12,
        heads=12,
        dec_dim=384,
        dec_depth=4,
        dec_heads=8,
        n_anat=4,
        mask_mode="anatomical",
        fusion="learnable",
        norm_pix_loss=False,
    ):
        super().__init__()
        assert mask_mode in {"random", "adaptive", "anatomical", "hard_anat"}
        self.mask_mode = mask_mode
        self.patch = patch
        self.in_ch = in_ch
        self.norm_pix_loss = norm_pix_loss

        self.patch_embed = PatchEmbed(img_size, patch, in_ch, dim)
        N = self.patch_embed.n_patches
        self.register_buffer("pos", sincos_pos_embed(dim, self.patch_embed.grid)[None])

        self.blocks = nn.ModuleList([TransformerBlock(dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)

        if mask_mode in {"adaptive", "anatomical"}:
            self.sampler = TokenSampler(
                dim, n_anat, fusion=fusion, use_anat=(mask_mode == "anatomical")
            )
        else:
            self.sampler = None

        # Decoder (Sec. 3.4.4 part 2)
        self.dec_embed = nn.Linear(dim, dec_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dec_dim))
        self.register_buffer("dec_pos", sincos_pos_embed(dec_dim, self.patch_embed.grid)[None])
        self.dec_blocks = nn.ModuleList(
            [TransformerBlock(dec_dim, dec_heads) for _ in range(dec_depth)]
        )
        self.dec_norm = nn.LayerNorm(dec_dim)
        self.dec_pred = nn.Linear(dec_dim, patch * patch * in_ch)

        self.apply(self._init)
        nn.init.normal_(self.mask_token, std=0.02)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    # ---------------------------------------------------------------------------------
    # patch <-> image
    # ---------------------------------------------------------------------------------
    def patchify(self, x):
        """[B, C, H, W] -> [B, N, patch*patch*C]"""
        p, c = self.patch, self.in_ch
        B, _, H, W = x.shape
        gh, gw = H // p, W // p
        return (
            x.reshape(B, c, gh, p, gw, p)
            .permute(0, 2, 4, 3, 5, 1)      # [B, gh, gw, p, p, c]
            .reshape(B, gh * gw, p * p * c)
        )

    def unpatchify(self, t):
        p, c = self.patch, self.in_ch
        B, N, _ = t.shape
        g = int(math.sqrt(N))
        return (
            t.reshape(B, g, g, p, p, c)
            .permute(0, 5, 1, 3, 2, 4)      # [B, c, gh, p, gw, p]
            .reshape(B, c, g * p, g * p)
        )

    # ---------------------------------------------------------------------------------
    # Masking (Sec. 3.4.3 part 3, Eq. 3.15-3.16)
    # ---------------------------------------------------------------------------------
    def sample_visible(self, x, anat, mask_ratio):
        """
        Returns (ids_keep [B, Nv], mask [B, N] with 1 = masked, probs [B, N] or None).

        Visible tokens are drawn *without replacement* from P via multinomial sampling,
        with N_v = N * (1 - p).
        """
        B, N, _ = x.shape
        Nv = max(1, int(round(N * (1.0 - mask_ratio))))
        probs = None

        if self.mask_mode == "random":
            ids_keep = torch.rand(B, N, device=x.device).argsort(dim=1)[:, :Nv]

        elif self.mask_mode == "hard_anat":
            # Fixed rule: keep the Nv patches with the strongest anatomical response.
            # Deterministic and non-learnable, mirroring hard prior-based masking.
            score = anat.mean(dim=-1) + 1e-4 * torch.rand_like(anat[..., 0])
            ids_keep = score.argsort(dim=1, descending=True)[:, :Nv]

        else:  # 'adaptive' or 'anatomical'
            probs = self.sampler(x, anat)                       # Eq. 3.3 / 3.14
            ids_keep = torch.multinomial(probs, Nv, replacement=False)  # Eq. 3.15

        mask = torch.ones(B, N, device=x.device)
        mask.scatter_(1, ids_keep, 0.0)                          # 1 = masked
        return ids_keep, mask, probs

    # ---------------------------------------------------------------------------------
    # Forward
    # ---------------------------------------------------------------------------------
    def forward_encoder(self, img, anat, mask_ratio):
        x = self.patch_embed(img) + self.pos                     # Eq. 3.1 + Eq. 3.17
        ids_keep, mask, probs = self.sample_visible(x, anat, mask_ratio)

        # Gather visible tokens X_v
        xv = torch.gather(x, 1, ids_keep[..., None].expand(-1, -1, x.shape[-1]))
        for blk in self.blocks:                                  # Eq. 3.4 / 3.18
            xv = blk(xv)
        return self.norm(xv), mask, ids_keep, probs

    def forward_decoder(self, fv, ids_keep, N):
        """Eq. 3.19-3.20 and Eq. 3.5: scatter visible latents back, fill with mask tokens."""
        B = fv.shape[0]
        fv = self.dec_embed(fv)
        full = self.mask_token.expand(B, N, -1).clone()
        full = full.scatter(1, ids_keep[..., None].expand(-1, -1, fv.shape[-1]), fv)
        full = full + self.dec_pos                               # Eq. 3.20
        for blk in self.dec_blocks:
            full = blk(full)
        return self.dec_pred(self.dec_norm(full))                # [B, N, p*p*C]

    def forward(self, img, anat=None, mask_ratio=0.75):
        if anat is None:
            anat = torch.zeros(img.shape[0], self.patch_embed.n_patches, 4, device=img.device)

        fv, mask, ids_keep, probs = self.forward_encoder(img, anat, mask_ratio)
        pred = self.forward_decoder(fv, ids_keep, self.patch_embed.n_patches)

        target = self.patchify(img)
        if self.norm_pix_loss:
            mu = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mu) / (var + 1e-6) ** 0.5

        # Per-token reconstruction error
        per_tok = ((pred - target) ** 2).mean(dim=-1)             # [B, N]

        # Eq. 3.21: L_R averaged over masked tokens only
        loss_recon = (per_tok * mask).sum() / mask.sum().clamp(min=1)

        # Eq. 3.22: L_S = -sum_{i in I_m} p_i * L_R(i)
        # Gradients are detached from the encoder-decoder (Sec. 3.5.2) so the sampler
        # is optimized independently of the reconstruction network.
        loss_sampler = torch.zeros((), device=img.device)
        if probs is not None:
            r = (per_tok * mask).detach()
            loss_sampler = -(probs * r).sum(dim=1).mean()

        return {
            "loss_recon": loss_recon,
            "loss_sampler": loss_sampler,
            "pred": pred,
            "mask": mask,
            "probs": probs,
        }


def build_model(args):
    return MaskedAutoencoder(
        img_size=args.img_size,
        patch=args.patch,
        in_ch=4,
        dim=args.dim,
        depth=args.depth,
        heads=args.heads,
        dec_dim=args.dec_dim,
        dec_depth=args.dec_depth,
        mask_mode=args.mask_mode,
        fusion=args.fusion,
        norm_pix_loss=args.norm_pix_loss,
    )
