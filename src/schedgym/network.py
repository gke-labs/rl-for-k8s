"""Joint pod/node transformer policy+value network (DESIGN.md sec 7).

Pods and nodes share one token sequence and attend to each other (and to
each other's kind) in every layer, so the policy has a chance to
coordinate simultaneous per-pod decisions (e.g. two mutually anti-affine
pods preferring different nodes) despite actions being emitted in one
shot rather than autoregressively -- see DESIGN.md sec 5 for why that
coordination has to live in the shared representation.

Operates on a single (unbatched) `Encoded` cluster; wrap with
`jax.vmap(model.apply, in_axes=(None, 0))` to run over a training batch.
"""

from __future__ import annotations

import flax.linen as nn
import jax.numpy as jnp

from schedgym.encoding import Encoded, NUM_HASH_BUCKETS

NEG_INF = -1e9


def _embed_bag(embed_layer: nn.Embed, ids: jnp.ndarray) -> jnp.ndarray:
    """Sum token embeddings over a bag of hashed ids, ignoring pad (id 0)."""
    emb = embed_layer(ids)  # [..., L, D]
    mask = (ids > 0)[..., None].astype(jnp.float32)
    return jnp.sum(emb * mask, axis=-2)  # [..., D]


class TransformerBlock(nn.Module):
    d_model: int
    num_heads: int
    ff_mult: int = 4

    @nn.compact
    def __call__(self, x: jnp.ndarray, valid_mask: jnp.ndarray) -> jnp.ndarray:
        attn_mask = (valid_mask[None, :] & valid_mask[:, None])[None, :, :]  # broadcast over heads

        h = nn.LayerNorm()(x)
        attn_out = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads, qkv_features=self.d_model, out_features=self.d_model
        )(h, h, mask=attn_mask)
        x = x + attn_out

        h = nn.LayerNorm()(x)
        ff = nn.Dense(self.d_model * self.ff_mult)(h)
        ff = nn.gelu(ff)
        ff = nn.Dense(self.d_model)(ff)
        x = x + ff

        return x * valid_mask[:, None]  # keep padding tokens at exactly zero through the residual stream


class SchedulerPolicy(nn.Module):
    d_model: int = 128
    num_heads: int = 4
    num_layers: int = 3
    embed_dim: int = 32

    @nn.compact
    def __call__(self, encoded: Encoded) -> tuple[jnp.ndarray, jnp.ndarray]:
        label_embed = nn.Embed(NUM_HASH_BUCKETS, self.embed_dim, name="label_embed")

        pod_tokens = (
            nn.Dense(self.d_model, name="pod_numeric_proj")(encoded.pod_numeric)
            + nn.Dense(self.d_model, name="pod_labels_proj")(_embed_bag(label_embed, encoded.pod_labels))
            + nn.Dense(self.d_model, name="pod_selector_proj")(_embed_bag(label_embed, encoded.pod_selector))
            + nn.Dense(self.d_model, name="pod_tol_proj")(_embed_bag(label_embed, encoded.pod_tolerations))
            + self.param("pod_type_embed", nn.initializers.normal(0.02), (self.d_model,))
        )

        node_tokens = (
            nn.Dense(self.d_model, name="node_numeric_proj")(encoded.node_numeric)
            + nn.Dense(self.d_model, name="node_labels_proj")(_embed_bag(label_embed, encoded.node_labels))
            + nn.Dense(self.d_model, name="node_taints_proj")(_embed_bag(label_embed, encoded.node_taints))
            + self.param("node_type_embed", nn.initializers.normal(0.02), (self.d_model,))
        )

        tokens = jnp.concatenate([pod_tokens, node_tokens], axis=0)  # [P+N, d]
        valid_mask = jnp.concatenate([encoded.pod_mask, encoded.node_mask], axis=0)

        for _ in range(self.num_layers):
            tokens = TransformerBlock(self.d_model, self.num_heads)(tokens, valid_mask)

        num_pods = encoded.pod_mask.shape[0]
        pod_repr, node_repr = tokens[:num_pods], tokens[num_pods:]

        query = nn.Dense(self.embed_dim, name="policy_query")(pod_repr)  # [P, e]
        key = nn.Dense(self.embed_dim, name="policy_key")(node_repr)  # [N, e]
        logits = jnp.einsum("pe,ne->pn", query, key) / jnp.sqrt(self.embed_dim)

        # Padding-pod rows and fully-infeasible rows end up all -inf, which
        # softmaxes to a harmless uniform distribution (not NaN); the reward
        # is what teaches the policy such rows are bad, not the mask itself.
        logits = jnp.where(encoded.action_mask, logits, NEG_INF)

        pooled = jnp.sum(tokens * valid_mask[:, None], axis=0) / jnp.maximum(jnp.sum(valid_mask), 1.0)
        value = nn.Dense(self.d_model, name="value_hidden")(pooled)
        value = nn.gelu(value)
        value = nn.Dense(1, name="value_out")(value)[0]

        return logits, value
