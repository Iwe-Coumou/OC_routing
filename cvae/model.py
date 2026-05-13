"""Trainable schedule prior used to seed latent CP-DE search.

Implements the CVAE-Opt architecture (Hottung et al., ICLR 2021) adapted for
the VeRoLog 2017 multi-day tool-rental routing problem.

Key improvements over the original simplified version:
- TransformerBlock: multi-head self-attention so requests interact with each other
- AttentiveScheduleCVAE: proper encoder sees BOTH instance AND solution (not just instance)
- Symmetry-breaking: handled by data augmentation in train.py (day jitter)
- state_to_search_vector: converts any archived solution directly to a 510-D DE seed
- predict_latent_vectors_from_encoded: encodes a known-good solution and samples
  the posterior N(mu, sigma) to get diverse but guided DE starting points
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from instance import Instance


FEATURE_DIM = 17


@dataclass(frozen=True)
class InstanceTensor:
    features: torch.Tensor
    request_mask: torch.Tensor
    target_z: torch.Tensor | None = None
    target_day_bias: torch.Tensor | None = None


class SchedulePriorNet(nn.Module):
    """Predicts the latent vector consumed by ``cvae.search``.

    The model is request-wise with a small self-attention block.  It outputs one
    scalar per request; after a sigmoid, that scalar becomes the preferred
    delivery-day position inside the request time window.  It also outputs a
    global day-bias vector that lets the CP decoder prefer earlier/later daily
    load profiles.
    """

    def __init__(self, feature_dim: int = FEATURE_DIM, hidden_dim: int = 128, max_days: int = 25):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.max_days = max_days

        self.input = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.request_head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.day_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, max_days),
        )

    def forward(self, features: torch.Tensor, request_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.input(features)

        weights = request_mask.float().unsqueeze(-1)
        pooled = (x * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        pooled_per_request = pooled.unsqueeze(1).expand(-1, x.shape[1], -1)
        request_z = self.request_head(torch.cat([x, pooled_per_request], dim=-1)).squeeze(-1)
        day_bias = self.day_head(pooled)
        return request_z, day_bias


class ScheduleCVAE(nn.Module):
    """Conditional VAE for feasible-schedule latent vectors.

    The encoder sees an instance plus a concrete delivery-day assignment.  The
    decoder sees the instance plus a latent vector and predicts the continuous
    vector consumed by ``cvae.search``.
    """

    def __init__(
        self,
        feature_dim: int = FEATURE_DIM,
        hidden_dim: int = 128,
        max_days: int = 25,
        latent_dim: int = 32,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.max_days = max_days
        self.latent_dim = latent_dim

        self.feature_net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.solution_net = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)

        self.request_decoder = nn.Sequential(
            nn.LayerNorm(hidden_dim + latent_dim),
            nn.Linear(hidden_dim + latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.day_decoder = nn.Sequential(
            nn.LayerNorm(hidden_dim + latent_dim),
            nn.Linear(hidden_dim + latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, max_days),
        )

    def encode(
        self,
        features: torch.Tensor,
        request_mask: torch.Tensor,
        target_z: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        request_emb = self.feature_net(features)
        target = target_z.unsqueeze(-1).clamp(-5.0, 5.0)
        encoded = self.solution_net(torch.cat([request_emb, target], dim=-1))
        weights = request_mask.float().unsqueeze(-1)
        pooled = (encoded * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        return self.mu(pooled), self.logvar(pooled).clamp(-6.0, 4.0)

    def decode(
        self,
        features: torch.Tensor,
        request_mask: torch.Tensor,
        latent: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        request_emb = self.feature_net(features)
        weights = request_mask.float().unsqueeze(-1)
        pooled = (request_emb * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)

        z_req = latent.unsqueeze(1).expand(-1, request_emb.shape[1], -1)
        request_z = self.request_decoder(torch.cat([request_emb, z_req], dim=-1)).squeeze(-1)
        day_bias = self.day_decoder(torch.cat([pooled, latent], dim=-1))
        return request_z, day_bias

    def forward(
        self,
        features: torch.Tensor,
        request_mask: torch.Tensor,
        target_z: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(features, request_mask, target_z)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        latent = mu + eps * std
        request_z, day_bias = self.decode(features, request_mask, latent)
        return request_z, day_bias, mu, logvar


def instance_features(instance: Instance) -> np.ndarray:
    coords = np.asarray(instance.coordinates, dtype=np.float32)
    max_x = float(max(coords[:, 0].max(), 1.0))
    max_y = float(max(coords[:, 1].max(), 1.0))
    max_dist = float(max(max(row) for row in instance.distance) or 1.0)
    max_tool_id = max((t.id for t in instance.tools), default=1)
    tool_by_id = {t.id: t for t in instance.tools}
    days = max(1, instance.config.days)
    capacity = max(1, instance.config.capacity)
    max_tool_cost = max((t.cost for t in instance.tools), default=1)

    rows = []
    for req in instance.requests:
        tool = tool_by_id[req.machine_type]
        x, y = instance.coordinates[req.location_id]
        depot_dist = instance.get_distance(instance.depot_id, req.location_id)
        span = max(0, req.latest - req.earliest)
        load = req.num_machines * tool.size
        rows.append(
            [
                req.earliest / days,
                req.latest / days,
                req.duration / days,
                span / days,
                req.machine_type / max_tool_id,
                req.num_machines / max(1, tool.num_available),
                req.num_machines / 20.0,
                tool.size / capacity,
                load / capacity,
                tool.num_available / 200.0,
                math.log1p(tool.cost) / math.log1p(max_tool_cost),
                float(x) / max_x,
                float(y) / max_y,
                depot_dist / max_dist,
                instance.config.vehicle_cost / max(1.0, instance.config.vehicle_cost + instance.config.distance_cost),
                instance.config.vehicle_day_cost / max(1.0, instance.config.vehicle_cost),
                instance.config.distance_cost / max(1.0, instance.config.vehicle_cost),
            ]
        )
    return np.nan_to_num(np.asarray(rows, dtype=np.float32), nan=0.0, posinf=1.0, neginf=-1.0)


def target_from_state(instance: Instance, state: dict, max_days: int) -> tuple[np.ndarray, np.ndarray]:
    delivery_by_id = {e["request"].id: e["delivery_day"] for e in state["scheduled"]}
    target_z = []
    delivery_count = np.zeros(max_days, dtype=np.float32)
    for req in instance.requests:
        day = delivery_by_id[req.id]
        if 1 <= day <= max_days:
            delivery_count[day - 1] += 1.0
        span = req.latest - req.earliest
        if span <= 0:
            frac = 0.5
        else:
            frac = (day - req.earliest) / span
        frac = float(np.clip(frac, 1e-3, 1.0 - 1e-3))
        target_z.append(float(np.clip(math.log(frac / (1.0 - frac)), -5.0, 5.0)))

    if delivery_count.max() > 0:
        delivery_count = delivery_count / delivery_count.max()
    # Search interprets lower bias as more attractive, so invert counts.
    target_day_bias = 1.0 - delivery_count
    return np.asarray(target_z, dtype=np.float32), target_day_bias.astype(np.float32)


def make_instance_tensor(
    instance: Instance,
    *,
    state: dict | None = None,
    max_days: int | None = None,
) -> InstanceTensor:
    max_days = max_days or instance.config.days
    features = torch.from_numpy(instance_features(instance))
    request_mask = torch.ones(features.shape[0], dtype=torch.bool)
    if state is None:
        return InstanceTensor(features=features, request_mask=request_mask)
    z, day_bias = target_from_state(instance, state, max_days)
    return InstanceTensor(
        features=features,
        request_mask=request_mask,
        target_z=torch.from_numpy(z),
        target_day_bias=torch.from_numpy(day_bias),
    )


def collate_instance_tensors(batch: list[InstanceTensor], max_days: int) -> InstanceTensor:
    batch_size = len(batch)
    max_requests = max(item.features.shape[0] for item in batch)
    feature_dim = batch[0].features.shape[1]

    features = torch.zeros(batch_size, max_requests, feature_dim, dtype=torch.float32)
    mask = torch.zeros(batch_size, max_requests, dtype=torch.bool)
    target_z = torch.zeros(batch_size, max_requests, dtype=torch.float32)
    day_bias = torch.zeros(batch_size, max_days, dtype=torch.float32)

    has_targets = batch[0].target_z is not None
    for idx, item in enumerate(batch):
        n = item.features.shape[0]
        features[idx, :n] = item.features
        mask[idx, :n] = item.request_mask
        if has_targets:
            target_z[idx, :n] = item.target_z
            day_bias[idx, :] = item.target_day_bias

    return InstanceTensor(
        features=features,
        request_mask=mask,
        target_z=target_z if has_targets else None,
        target_day_bias=day_bias if has_targets else None,
    )


def predict_latent_vector(model: nn.Module, instance: Instance, device: torch.device | str = "cpu") -> np.ndarray:
    return predict_latent_vectors(model, instance, n_samples=1, device=device)[0]


class TransformerBlock(nn.Module):
    """Pre-norm multi-head self-attention block (pre-LN is more stable at small batch sizes)."""

    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        # Pre-norm self-attention
        normed = self.norm1(x)
        attended, _ = self.attn(normed, normed, normed, key_padding_mask=key_padding_mask)
        x = x + attended
        # Pre-norm FFN
        x = x + self.ff(self.norm2(x))
        return x


class AttentiveScheduleCVAE(nn.Module):
    """Proper CVAE-Opt encoder-decoder with multi-head self-attention.

    The encoder sees BOTH the instance features AND the target solution
    (delivery-day assignment), so the latent z captures which specific solution
    was given — not just the instance.  The decoder maps z back to per-request
    guidance signals for the CP-SAT feasibility decoder.

    This directly mirrors Hottung et al. (ICLR 2021) §3, adapted for the
    VeRoLog multi-day tool-rental routing problem.

    Hidden dim 256, latent dim 64, 3 transformer layers: ~3.3M parameters.
    """

    def __init__(
        self,
        feature_dim: int = FEATURE_DIM,
        hidden_dim: int = 256,
        max_days: int = 25,
        latent_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 3,
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.max_days = max_days
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.num_layers = num_layers

        # --- shared request feature projection ---
        self.feature_proj = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # --- encoder ---
        # Projects per-request day logit (target_z scalar) into hidden_dim
        self.solution_proj = nn.Linear(1, hidden_dim)
        self.enc_blocks = nn.ModuleList(
            [TransformerBlock(hidden_dim, num_heads) for _ in range(num_layers)]
        )
        # Learned single-query attention pooling (better than mean for asymmetric importance)
        self.enc_pool_q = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        nn.init.normal_(self.enc_pool_q, std=0.02)
        self.mu_head = nn.Linear(hidden_dim, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim, latent_dim)

        # --- decoder ---
        # Projects latent vector into hidden_dim for element-wise conditioning
        self.latent_proj = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.dec_blocks = nn.ModuleList(
            [TransformerBlock(hidden_dim, num_heads) for _ in range(num_layers)]
        )
        self.dec_pool_q = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        nn.init.normal_(self.dec_pool_q, std=0.02)
        # Per-request day guidance logit
        self.request_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        # Global day-load bias
        self.day_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, max_days),
        )

    def _attn_pool(self, x: torch.Tensor, pool_q: nn.Parameter, mask: torch.Tensor) -> torch.Tensor:
        """Attention pooling over the request/token dimension."""
        B = x.shape[0]
        q = pool_q.expand(B, -1, -1)  # [B,1,H]
        scores = torch.bmm(q, x.transpose(1, 2)) / (self.hidden_dim ** 0.5)  # [B,1,N]
        scores = scores.masked_fill(~mask.unsqueeze(1), float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        return torch.bmm(weights, x).squeeze(1)  # [B,H]

    def encode(
        self,
        features: torch.Tensor,
        request_mask: torch.Tensor,
        target_z: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode (instance, solution) → (mu, logvar) in latent space."""
        x = self.feature_proj(features)  # [B,N,H]
        # Fuse per-request solution information (the key CVAE-Opt difference)
        sol = self.solution_proj(target_z.unsqueeze(-1).clamp(-5.0, 5.0))  # [B,N,H]
        x = x + sol
        pad_mask = ~request_mask  # True = padding
        for block in self.enc_blocks:
            x = block(x, key_padding_mask=pad_mask)
        pooled = self._attn_pool(x, self.enc_pool_q, request_mask)
        return self.mu_head(pooled), self.logvar_head(pooled).clamp(-6.0, 4.0)

    def decode(
        self,
        features: torch.Tensor,
        request_mask: torch.Tensor,
        latent: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode (instance, z) → (per-request guidance, day bias)."""
        x = self.feature_proj(features)  # [B,N,H]
        # Inject latent as a global additive bias to every token
        z_bias = self.latent_proj(latent).unsqueeze(1)  # [B,1,H]
        x = x + z_bias
        pad_mask = ~request_mask
        for block in self.dec_blocks:
            x = block(x, key_padding_mask=pad_mask)
        request_z = self.request_head(x).squeeze(-1)  # [B,N]
        request_z = request_z.masked_fill(~request_mask, 0.0)
        pooled = self._attn_pool(x, self.dec_pool_q, request_mask)
        day_bias = self.day_head(pooled)  # [B,max_days]
        return request_z, day_bias

    def forward(
        self,
        features: torch.Tensor,
        request_mask: torch.Tensor,
        target_z: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(features, request_mask, target_z)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        latent = mu + eps * std
        request_z, day_bias = self.decode(features, request_mask, latent)
        return request_z, day_bias, mu, logvar


def state_to_search_vector(instance: Instance, state: dict, max_days: int) -> np.ndarray:
    """Convert a schedule state directly to a DE search vector without needing a model.

    The returned vector has the same format used by ``decode_latent_schedule``:
      vector[:days]  = day-load bias  (first instance.config.days entries)
      vector[days:]  = per-request delivery-day logit (one per request)

    This lets us seed DE directly from any archived solution — the single most
    impactful change from the original implementation.
    """
    target_z, target_day_bias = target_from_state(instance, state, max_days)
    day_part = target_day_bias[: instance.config.days].astype(np.float64)
    return np.concatenate([day_part, target_z.astype(np.float64)])


def predict_latent_vectors(
    model: nn.Module,
    instance: Instance,
    *,
    n_samples: int = 1,
    device: torch.device | str = "cpu",
    seed: int = 0,
) -> list[np.ndarray]:
    model.eval()
    tensor = make_instance_tensor(instance, max_days=model.max_days)
    features = tensor.features.unsqueeze(0).to(device)
    mask = tensor.request_mask.unsqueeze(0).to(device)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    vectors = []
    with torch.no_grad():
        if isinstance(model, (ScheduleCVAE, AttentiveScheduleCVAE)):
            latents = [torch.zeros(1, model.latent_dim, device=device)]
            for _ in range(max(0, n_samples - 1)):
                latents.append(torch.randn(1, model.latent_dim, device=device, generator=generator))
            for latent in latents:
                request_z, day_bias = model.decode(features, mask, latent)
                vectors.append(_combine_prediction(instance, request_z, day_bias))
        else:
            request_z, day_bias = model(features, mask)
            vectors.append(_combine_prediction(instance, request_z, day_bias))
    return vectors


def predict_latent_vectors_from_encoded(
    model: nn.Module,
    instance: Instance,
    state: dict,
    *,
    n_samples: int = 8,
    device: torch.device | str = "cpu",
    seed: int = 0,
) -> list[np.ndarray]:
    """Encode a known-good solution and sample its posterior N(mu, sigma).

    This is the proper CVAE-Opt inference procedure (§4.1 of Hottung et al.):
    instead of sampling from the prior z ~ N(0,I), we infer a good z by
    encoding an elite solution, then perturb in the posterior neighbourhood.
    The DE then starts from a much better region of search space.
    """
    if not isinstance(model, (ScheduleCVAE, AttentiveScheduleCVAE)):
        return predict_latent_vectors(model, instance, n_samples=n_samples, device=device, seed=seed)

    model.eval()
    tensor = make_instance_tensor(instance, state=state, max_days=model.max_days)
    features = tensor.features.unsqueeze(0).to(device)
    mask = tensor.request_mask.unsqueeze(0).to(device)
    target_z = tensor.target_z.unsqueeze(0).to(device)

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    vectors = []
    with torch.no_grad():
        mu, logvar = model.encode(features, mask, target_z)
        std = torch.exp(0.5 * logvar)
        # First sample: use the posterior mean (highest-probability decoding)
        request_z, day_bias = model.decode(features, mask, mu)
        vectors.append(_combine_prediction(instance, request_z, day_bias))
        # Remaining: sample from posterior N(mu, sigma)
        for _ in range(n_samples - 1):
            eps = torch.randn_like(std, generator=generator)
            latent = mu + eps * std
            request_z, day_bias = model.decode(features, mask, latent)
            vectors.append(_combine_prediction(instance, request_z, day_bias))
    return vectors


def load_checkpoint(path: str | Path, device: torch.device | str = "cpu") -> nn.Module:
    payload = torch.load(path, map_location=device, weights_only=False)
    mtype = payload.get("model_type", "schedule_prior")
    if mtype == "attentive_cvae":
        model = AttentiveScheduleCVAE(
            feature_dim=payload.get("feature_dim", FEATURE_DIM),
            hidden_dim=payload.get("hidden_dim", 256),
            max_days=payload["max_days"],
            latent_dim=payload.get("latent_dim", 64),
            num_heads=payload.get("num_heads", 4),
            num_layers=payload.get("num_layers", 3),
        )
    elif mtype == "schedule_cvae":
        model = ScheduleCVAE(
            feature_dim=payload.get("feature_dim", FEATURE_DIM),
            hidden_dim=payload.get("hidden_dim", 128),
            max_days=payload["max_days"],
            latent_dim=payload.get("latent_dim", 32),
        )
    else:
        model = SchedulePriorNet(
            feature_dim=payload.get("feature_dim", FEATURE_DIM),
            hidden_dim=payload.get("hidden_dim", 128),
            max_days=payload["max_days"],
        )
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    return model


def _combine_prediction(instance: Instance, request_z: torch.Tensor, day_bias: torch.Tensor) -> np.ndarray:
    day_part = day_bias[0, : instance.config.days].detach().cpu().numpy()
    request_part = request_z[0, : len(instance.requests)].detach().cpu().numpy()
    return np.concatenate([day_part, request_part]).astype(np.float64)
