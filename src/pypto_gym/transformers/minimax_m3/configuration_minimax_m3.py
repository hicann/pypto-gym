# coding=utf-8
# Copyright 2025 the HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# NOTICE: This file was modified by Huawei Technologies Co., Ltd. in 2026 to
# leverage pypto technology for fusing the MoE expert FFN via a grouped GEMM kernel.
#
"""MiniMax M3 text-backbone MoE configuration.

MiniMax-M3 ships as a vision-language model (the published ``config.json`` has
``model_type="minimax_m3_vl"``, architecture ``MiniMaxM3SparseForConditionalGeneration``)
whose **text backbone** is a Mixtral-style sparse-MoE decoder; the bundled
``configuration_minimax_m3_vl.py`` coerces the ``text_config`` sub-config to the
``minimax_m2`` / ``mixtral`` family (no native ``transformers`` arch). PyPTO only
accelerates that text MoE expert FFN via the fused grouped-GEMM kernel, so this
config captures the text backbone (``config.json["text_config"]``); the vision
tower / projector are loaded by the upstream VL code and are out of scope here.

Defaults below are the real MiniMax-M3 values taken from the published
``config.json`` (text_config), so ``MiniMaxM3Config()`` reproduces the shipped
text backbone:

    H=6144, moe_I=3072, E=128, top_k=4, sigmoid routing (+ bias), swigluoai act,
    60 layers (first 3 dense, rest MoE), 1M context, partial-rotary RoPE, qk-norm.
"""

__all__ = ["MiniMaxM3Config"]

from transformers.configuration_utils import PretrainedConfig


def _first_moe_layer(moe_layer_freq):
    """Index of the first MoE layer = ``first_k_dense_replace`` (leading dense block).

    ``moe_layer_freq`` is a per-layer 0/1 mask (0 = dense MLP, 1 = sparse MoE).
    MiniMax-M3 uses a leading run of dense layers; this returns that run length.
    Falls back to 0 if the mask is missing or all-MoE.
    """
    if not moe_layer_freq:
        return 0
    for i, flag in enumerate(moe_layer_freq):
        if flag:
            return i
    return len(moe_layer_freq)


class MiniMaxM3Config(PretrainedConfig):
    r"""Configuration for the MiniMax-M3 text-backbone MoE decoder.

    Inherits from [`PretrainedConfig`]; see it for the common arguments. Only the
    text backbone is modelled (the VL wrapper is handled upstream). The argument
    names mirror ``minimax_m27`` (the M2-family text backbone) plus the M3-specific
    MoE / sparse-attention fields, so the same PyPTO MoE patch applies.

    Args:
        vocab_size (`int`, *optional*, defaults to 200064): Vocabulary size.
        hidden_size (`int`, *optional*, defaults to 6144): Hidden dimension (H).
        intermediate_size (`int`, *optional*, defaults to 3072): Per-routed-expert
            MoE FFN width (a.k.a. ``moe_intermediate_size``, I).
        dense_intermediate_size (`int`, *optional*, defaults to 12288): FFN width of
            the leading dense (non-MoE) layers.
        shared_intermediate_size (`int`, *optional*, defaults to 3072): FFN width of
            the shared expert.
        num_hidden_layers (`int`, *optional*, defaults to 60): Decoder layers.
        num_attention_heads (`int`, *optional*, defaults to 64): Query heads.
        num_key_value_heads (`int`, *optional*, defaults to 4): KV heads (GQA).
        head_dim (`int`, *optional*, defaults to 128): Attention head dimension.
        hidden_act (`str`, *optional*, defaults to `"swigluoai"`): Expert/MLP
            activation. M3 uses the clamped GLU ``(up+1) * gate*sigmoid(alpha*gate)``
            parameterized by ``swiglu_alpha`` / ``swiglu_limit``.
        swiglu_alpha (`float`, *optional*, defaults to 1.702): swigluoai gate gain.
        swiglu_limit (`float`, *optional*, defaults to 7.0): swigluoai clamp limit.
        rms_norm_eps (`float`, *optional*, defaults to 1e-6): RMSNorm epsilon.
        use_gemma_norm (`bool`, *optional*, defaults to `True`): Gemma-style
            ``(1 + weight)`` RMSNorm scaling.
        max_position_embeddings (`int`, *optional*, defaults to 1048576): Max context.
        rope_theta (`float`, *optional*, defaults to 5000000): RoPE base period.
        rotary_dim (`int`, *optional*, defaults to 64): Rotary dimension (partial RoPE).
        partial_rotary_factor (`float`, *optional*, defaults to 0.5): rotary_dim/head_dim.
        use_qk_norm (`bool`, *optional*, defaults to `True`): RMSNorm on Q/K.
        qk_norm_type (`str`, *optional*, defaults to `"per_head"`): Q/K norm granularity.
        attention_output_gate (`bool`, *optional*, defaults to `False`): Output gate.
        num_local_experts (`int`, *optional*, defaults to 128): Routed experts (E).
        num_experts_per_tok (`int`, *optional*, defaults to 4): Routed top-k.
        n_shared_experts (`int`, *optional*, defaults to 1): Always-on shared experts.
        scoring_func (`str`, *optional*, defaults to `"sigmoid"`): Router scoring.
        use_routing_bias (`bool`, *optional*, defaults to `True`): Per-expert score bias
            (``e_score_correction_bias``).
        routed_scaling_factor (`float`, *optional*, defaults to 2.0): Scales the summed
            routed-expert output.
        norm_topk_prob (`bool`, *optional*, defaults to `True`): Normalize top-k weights.
        first_k_dense_replace (`int`, *optional*): Leading dense layers. Derived from
            ``moe_layer_freq`` when not given (M3: 3).
        moe_layer_freq (`list[int]`, *optional*): Per-layer dense(0)/MoE(1) mask.
        sparse_attention_config (`dict`, *optional*): MiniMax Sparse Attention (MSA)
            settings; carried through for the upstream attention, unused by the MoE kernel.
        output_router_logits (`bool`, *optional*, defaults to `False`): Return router logits.
        router_aux_loss_coef (`float`, *optional*, defaults to 0.001): Aux-loss factor.
    """

    model_type = "minimax_m3"
    keys_to_ignore_at_inference = ["past_key_values"]
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.block_sparse_moe.gate": "colwise_rep",
        "layers.*.block_sparse_moe.experts.*.w1": "colwise",
        "layers.*.block_sparse_moe.experts.*.w2": "rowwise",
        "layers.*.block_sparse_moe.experts.*.w3": "colwise",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    def __init__(
        self,
        vocab_size=200064,
        hidden_size=6144,
        intermediate_size=3072,
        dense_intermediate_size=12288,
        shared_intermediate_size=3072,
        num_hidden_layers=60,
        num_attention_heads=64,
        num_key_value_heads=4,
        head_dim=128,
        hidden_act="swigluoai",
        swiglu_alpha=1.702,
        swiglu_limit=7.0,
        rms_norm_eps=1e-6,
        use_gemma_norm=True,
        initializer_range=0.02,
        max_position_embeddings=1048576,
        rope_theta=5000000,
        rotary_dim=64,
        partial_rotary_factor=0.5,
        use_qk_norm=True,
        qk_norm_type="per_head",
        attention_output_gate=False,
        use_cache=True,
        rope_scaling=None,
        sliding_window=None,
        attention_dropout=0.0,
        pad_token_id=None,
        bos_token_id=200019,
        eos_token_id=200020,
        tie_word_embeddings=False,
        # MoE
        num_local_experts=128,
        num_experts_per_tok=4,
        n_shared_experts=1,
        scoring_func="sigmoid",
        use_routing_bias=True,
        routed_scaling_factor=2.0,
        norm_topk_prob=True,
        first_k_dense_replace=None,
        moe_layer_freq=None,
        sparse_attention_config=None,
        output_router_logits=False,
        router_aux_loss_coef=0.001,
        router_jitter_noise=0.0,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.dense_intermediate_size = dense_intermediate_size
        self.shared_intermediate_size = shared_intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim

        self.hidden_act = hidden_act
        self.swiglu_alpha = swiglu_alpha
        self.swiglu_limit = swiglu_limit
        self.rms_norm_eps = rms_norm_eps
        self.use_gemma_norm = use_gemma_norm
        self.initializer_range = initializer_range
        self.max_position_embeddings = max_position_embeddings

        self._init_rope(rope_theta, rotary_dim, head_dim, partial_rotary_factor, rope_scaling)
        self.sliding_window = sliding_window

        self.use_qk_norm = use_qk_norm
        self.qk_norm_type = qk_norm_type
        self.attention_output_gate = attention_output_gate
        self.use_cache = use_cache
        self.attention_dropout = attention_dropout

        # MoE configs
        self.num_local_experts = num_local_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.n_shared_experts = n_shared_experts
        self.scoring_func = scoring_func
        self.use_routing_bias = use_routing_bias
        self.routed_scaling_factor = routed_scaling_factor
        self.norm_topk_prob = norm_topk_prob
        self.moe_layer_freq = moe_layer_freq
        if first_k_dense_replace is None:
            first_k_dense_replace = _first_moe_layer(moe_layer_freq)
        self.first_k_dense_replace = first_k_dense_replace
        self.sparse_attention_config = sparse_attention_config or {}
        self.output_router_logits = output_router_logits
        self.router_aux_loss_coef = router_aux_loss_coef
        self.router_jitter_noise = router_jitter_noise

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    @classmethod
    def from_vl_config_dict(cls, config_dict):
        """Build the text-backbone config from a full MiniMax-M3 VL ``config.json``.

        Accepts the parsed VL config dict (the published ``config.json``) and pulls
        ``text_config`` (falling back to the dict itself if already a text config),
        dropping VL-only / loader keys that aren't ``__init__`` arguments.
        """
        text = dict(config_dict.get("text_config", config_dict))
        drop = ("architectures", "auto_map", "model_type", "torch_dtype", "dtype",
                "transformers_version", "quantization_config", "_pre_quantization_dtype")
        return cls(**{k: v for k, v in text.items() if k not in drop})

    def _init_rope(self, rope_theta, rotary_dim, head_dim, partial_rotary_factor, rope_scaling):
        """Partial-rotary RoPE: ``rotary_dim`` is the rotated slice of ``head_dim``."""
        self.rope_theta = rope_theta
        self.rotary_dim = rotary_dim
        if head_dim:
            self.partial_rotary_factor = rotary_dim / head_dim
        else:
            self.partial_rotary_factor = partial_rotary_factor
        self.rope_scaling = rope_scaling
