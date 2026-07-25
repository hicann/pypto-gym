# coding=utf-8
# Adapted from: https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct/blob/main/configuration_kimi.py
#
# NOTICE: This file was modified by Huawei Technologies Co., Ltd. in 2026 to
# leverage PyPTO for operator fusion on Ascend NPU (config grouping for the KDA
# PyPTO integration).
from dataclasses import dataclass
from typing import Optional

from transformers.configuration_utils import PretrainedConfig


@dataclass
class MoeConfig:
    """The MoE-related config fields, grouped so the setter stays at <=5 args."""
    num_experts: Optional[int]
    num_experts_per_token: Optional[int]
    moe_renormalize: bool
    num_shared_experts: int
    routed_scaling_factor: float
    moe_router_activation_func: str
    moe_intermediate_size: Optional[int]
    first_k_dense_replace: int
    moe_layer_freq: int
    use_grouped_topk: bool
    num_expert_group: int
    topk_group: int


class KimiLinearConfig(PretrainedConfig):
    model_type = "kimi_linear"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        model_type="kimi_linear",
        vocab_size=163840,
        hidden_size=4096,
        head_dim=None,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=None,
        hidden_act="silu",
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        rope_theta=10000.0,
        rope_scaling=None,
        tie_word_embeddings=False,
        moe_intermediate_size: Optional[int] = None,
        moe_renormalize: bool = True,
        moe_router_activation_func: str = "sigmoid",
        num_experts: Optional[int] = None,
        num_experts_per_token: Optional[int] = None,
        num_shared_experts: int = 0,
        routed_scaling_factor: float = 1.0,
        first_k_dense_replace: int = 0,
        moe_layer_freq: int = 1,
        use_grouped_topk: bool = True,
        num_expert_group: int = 1,
        topk_group: int = 1,
        q_lora_rank: Optional[int] = None,
        kv_lora_rank: Optional[int] = None,
        qk_nope_head_dim: Optional[int] = None,
        qk_rope_head_dim: Optional[int] = None,
        v_head_dim: Optional[int] = None,
        mla_use_nope: Optional[bool] = False,
        num_nextn_predict_layers: int = 0,
        linear_attn_config: Optional[dict] = None,
        **kwargs,
    ):
        self.model_type = model_type
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.head_dim = (
            head_dim if head_dim is not None else hidden_size // num_attention_heads
        )
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling

        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.mla_use_nope = mla_use_nope
        self._init_moe_config(MoeConfig(
            num_experts=num_experts,
            num_experts_per_token=num_experts_per_token,
            moe_renormalize=moe_renormalize,
            num_shared_experts=num_shared_experts,
            routed_scaling_factor=routed_scaling_factor,
            moe_router_activation_func=moe_router_activation_func,
            moe_intermediate_size=moe_intermediate_size,
            first_k_dense_replace=first_k_dense_replace,
            moe_layer_freq=moe_layer_freq,
            use_grouped_topk=use_grouped_topk,
            num_expert_group=num_expert_group,
            topk_group=topk_group,
        ))
        self.num_nextn_predict_layers = num_nextn_predict_layers

        self.linear_attn_config = self._validated_linear_attn_config(linear_attn_config)

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    @property
    def is_mla(self):
        return (
            self.q_lora_rank is not None
            or self.kv_lora_rank is not None
            or self.qk_nope_head_dim is not None
            or self.qk_rope_head_dim is not None
            or self.v_head_dim is not None
            or self.mla_use_nope is True
        )

    @property
    def is_moe(self):
        return self.num_experts is not None

    @property
    def is_linear_attn(self) -> bool:
        return not (
            self.linear_attn_config is None
            or (
                isinstance(self.linear_attn_config, dict)
                and self.linear_attn_config["kda_layers"] is not None
                and not self.linear_attn_config["kda_layers"]
            )
        )

    def is_kda_layer(self, layer_idx: int):
        return (
            self.linear_attn_config is not None
            and (layer_idx + 1) in self.linear_attn_config["kda_layers"]
        )

    def _validated_linear_attn_config(self, linear_attn_config):
        """Validate the linear_attn_config dict (kda/full-attn layer keys) and return it."""
        if linear_attn_config is not None:
            if linear_attn_config["kda_layers"] is None:
                raise ValueError("linear_attn_config['kda_layers'] must not be None")
            if linear_attn_config["full_attn_layers"] is None:
                raise ValueError("linear_attn_config['full_attn_layers'] must not be None")
        return linear_attn_config

    def _init_moe_config(self, moe):
        """Set the MoE-related config attributes from a ``MoeConfig`` (same attrs as before)."""
        self.num_experts = moe.num_experts
        self.num_experts_per_token = moe.num_experts_per_token
        self.moe_renormalize = moe.moe_renormalize
        self.num_shared_experts = moe.num_shared_experts
        self.routed_scaling_factor = moe.routed_scaling_factor
        self.moe_router_activation_func = moe.moe_router_activation_func
        if self.moe_router_activation_func not in ("softmax", "sigmoid"):
            raise ValueError(
                "moe_router_activation_func must be 'softmax' or 'sigmoid', "
                f"got {self.moe_router_activation_func}")
        self.moe_intermediate_size = moe.moe_intermediate_size
        self.first_k_dense_replace = moe.first_k_dense_replace
        self.moe_layer_freq = moe.moe_layer_freq
        self.use_grouped_topk = moe.use_grouped_topk
        self.num_expert_group = moe.num_expert_group
        self.topk_group = moe.topk_group
