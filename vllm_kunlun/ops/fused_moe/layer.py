# """layer.py"""

# from contextlib import nullcontext
# from typing import Callable, Optional, Union, get_args

# import torch
# from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
# from vllm.model_executor.layers.fused_moe import FusedMoE
# from vllm.model_executor.layers.fused_moe.layer import UnquantizedFusedMoEMethod

# def apply(
#         self,
#         layer: torch.nn.Module,
#         x: torch.Tensor,
#         router_logits: torch.Tensor,
#         top_k: int,
#         renormalize: bool,
#         use_grouped_topk: bool = False,
#         topk_group: Optional[int] = None,
#         num_expert_group: Optional[int] = None,
#         global_num_experts: int = -1,
#         expert_map: Optional[torch.Tensor] = None,
#         custom_routing_function: Optional[Callable] = None,
#         scoring_func: str = "softmax",
#         routed_scaling_factor: float = 1.0,
#         e_score_correction_bias: Optional[torch.Tensor] = None,
#         apply_router_weight_on_input: bool = False,
#         activation: str = "silu",
#         enable_eplb: bool = False,
#         expert_load_view: Optional[torch.Tensor] = None,
#         logical_to_physical_map: Optional[torch.Tensor] = None,
#         logical_replica_count: Optional[torch.Tensor] = None,
#     ) -> torch.Tensor:
#         """apply"""
#         if enable_eplb:
#             raise NotImplementedError(
#                 "EPLB not supported for `UnquantizedFusedMoEMethod` yet.")
        
#         """forward_kunlun"""
#         from vllm_kunlun.ops._kunlun_ops import KunlunOps as ops
#         if self.moe.use_ep:
#             return ops.fused_moe_ep(x,
#                              layer.w13_weight,
#                              layer.w2_weight,
#                              router_logits,
#                              self.moe.ep_rank,
#                              top_k,
#                              renormalize=renormalize,
#                              inplace=True,
#                              use_grouped_topk=use_grouped_topk,
#                              num_expert_group=num_expert_group,
#                              topk_group=topk_group)
#         else:
#             return ops.fused_moe(x,
#                              layer.w13_weight,
#                              layer.w2_weight,
#                              router_logits,
#                              self.moe.ep_rank,
#                              top_k,
#                              renormalize=renormalize,
#                              inplace=True,
#                              use_grouped_topk=use_grouped_topk,
#                              num_expert_group=num_expert_group,
#                              topk_group=topk_group,
#                              scoring_func=scoring_func,
#                              e_score_correction_bias=e_score_correction_bias,
#                              w1_bias=getattr(layer, 'w13_bias', None),
#                              w2_bias=getattr(layer, 'w2_bias', None),
#                              )

# UnquantizedFusedMoEMethod.apply = apply

# class VllmFusedMoE(FusedMoE):
#     def __init__(
#         self,
#         num_experts: int,  # Global number of experts
#         top_k: int,
#         hidden_size: int,
#         intermediate_size: int,
#         params_dtype: Optional[torch.dtype] = None,
#         reduce_results: bool = False,
#         renormalize: bool = True,
#         use_grouped_topk: bool = False,
#         num_expert_group: Optional[int] = 0,
#         topk_group: Optional[int] = 0,
#         quant_config: Optional[QuantizationConfig] = None,
#         tp_size: Optional[int] = None,
#         ep_size: Optional[int] = None,
#         dp_size: Optional[int] = None,
#         prefix: str = "",
#         custom_routing_function: Optional[Callable] = None,
#         scoring_func: str = "softmax",
#         routed_scaling_factor: float = 1.0,
#         e_score_correction_bias: Optional[torch.Tensor] = None,
#         apply_router_weight_on_input: bool = False,
#         activation: str = "silu",
#         enable_eplb: bool = False,
#         num_redundant_experts: int = 0,
#         has_bias: bool = False,
#         is_sequence_parallel=False,
#         zero_expert_num: Optional[int] = 0,
#         zero_expert_type: Optional[str] = None,
#     ):
#         super().__init__(
#             num_experts=num_experts,  # Global number of experts
#             top_k=top_k,
#             hidden_size=hidden_size,
#             intermediate_size=intermediate_size,
#             params_dtype=params_dtype,
#             reduce_results=reduce_results,
#             renormalize=renormalize,
#             use_grouped_topk=use_grouped_topk,
#             num_expert_group=num_expert_group,
#             topk_group=topk_group,
#             quant_config=quant_config,
#             tp_size=tp_size,
#             ep_size=ep_size,
#             dp_size=dp_size,
#             prefix=prefix,
#             custom_routing_function=custom_routing_function,
#             scoring_func=scoring_func,
#             routed_scaling_factor=routed_scaling_factor,
#             e_score_correction_bias=e_score_correction_bias,
#             apply_router_weight_on_input=apply_router_weight_on_input,
#             activation=activation,
#             enable_eplb=enable_eplb,
#             num_redundant_experts=num_redundant_experts,
#             has_bias=has_bias,
#             is_sequence_parallel=is_sequence_parallel,
#             zero_expert_num=zero_expert_num,
#             zero_expert_type=zero_expert_type)
#         self.has_bias=has_bias
#         self.register_parameter("w13_bias", None)
#         self.register_parameter("w2_bias", None)

#     def forward_native(
#         self,
#         hidden_states: torch.Tensor,
#         router_logits: torch.Tensor,
#         linear_weights: torch.Tensor = None
#     ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
#         og_hidden_states = hidden_states.shape[-1]
#         if self.hidden_size != og_hidden_states:
#             hidden_states = F.pad(hidden_states,
#                                   (0, self.hidden_size - og_hidden_states),
#                                   mode='constant',
#                                   value=0.0)

#         if self.shared_experts is None:
#             if current_platform.is_tpu():
#                 # TODO: Once the OOM issue for the TPU backend is resolved, we
#                 # will switch to using the moe_forward custom op.
#                 fused_output = self.forward_impl(hidden_states, router_logits)
#                 assert not isinstance(fused_output, tuple)
#             else:
#                 fused_output = torch.ops.vllm.moe_forward(
#                     hidden_states, router_logits, self.layer_name)
#             return fused_output[..., :og_hidden_states]
#         else:
#             if current_platform.is_tpu():
#                 # TODO: Once the OOM issue for the TPU backend is resolved, we
#                 # will switch to using the moe_forward custom op.
#                 shared_output, fused_output = self.forward_impl(
#                     hidden_states, router_logits)
#             else:
#                 # shared_output, fused_output = torch.ops.vllm.moe_forward_shared(
#                 #     hidden_states, router_logits, self.layer_name)
#                 forward_context: ForwardContext = get_forward_context()
#                 self = forward_context.no_compile_layers[self.layer_name]
#                 assert self.shared_experts is not None
#                 shared_output, fused_output =  self.forward_impl(hidden_states, router_logits, linear_weights)
#                 # shared_output =  self.forward_impl(hidden_states, router_logits, linear_weights)
#             return (shared_output[..., :og_hidden_states],
#                     fused_output[..., :og_hidden_states])
#             # return shared_output[..., :og_hidden_states]

#     def forward(self, hidden_states: torch.Tensor,
#                 router_logits: torch.Tensor = None,
#                 linear_weights: torch.Tensor = None
#         )-> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
#         """forward"""
#         # TODO: Once the OOM issue for the TPU backend is resolved, we will
#         # switch to using the moe_forward custom op.
#         # if current_platform.is_tpu():
#         #     return self.forward_impl(hidden_states, router_logits)
#         # else:
#         #     forward_context: ForwardContext = get_forward_context()
#         #     self = forward_context.no_compile_layers[self.layer_name]
#         #     assert self.quant_method is not None
#         #     return self.forward_impl(hidden_states, router_logits, linear_weights)
#         #     # return torch.ops.vllm.moe_forward(hidden_states, router_logits,
#         #     #                                   self.layer_name)

#         return self.forward_native(hidden_states, router_logits, linear_weights)

#     def forward_impl(self, hidden_states: torch.Tensor,
#                      router_logits: torch.Tensor,
#                      linear_weights: torch.Tensor = None
#         )-> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
#         """forward_impl"""
#         assert self.quant_method is not None

#         self.ensure_moe_quant_config()

#         # Route to the chunked forward path using the FlashInfer Cutlass kernel
#         # only when data parallelism (DP) is enabled.
#         _use_flashinfer_cutlass_kernels = (self.dp_size > 1 and
#                                            self.use_flashinfer_cutlass_kernels)

#         if (self.moe_parallel_config.use_pplx_kernels
#                 or self.moe_parallel_config.use_deepep_ll_kernels
#                 or _use_flashinfer_cutlass_kernels):
#             return self.forward_impl_chunked(hidden_states, router_logits)

#         do_naive_dispatch_combine: bool = (
#             self.dp_size > 1
#             and not self.moe_parallel_config.use_deepep_ht_kernels
#             and not self.moe_config.use_flashinfer_cutlass_kernels)

#         # If there are shared experts but we are not using a modular kernel, the
#         # shared experts must be called here
#         if (not isinstance(self.quant_method.fused_experts,
#                            FusedMoEModularKernel)
#                 and self.shared_experts is not None):
#             shared_output = self.shared_experts(hidden_states)
#         else:
#             shared_output = None

#         ctx = get_forward_context()
#         sp_ctx = ctx.dp_metadata.sp_local_sizes(
#             self.sp_size) if ctx.dp_metadata else nullcontext()

#         with sp_ctx:
#             if do_naive_dispatch_combine:
#                 hidden_states, router_logits = get_ep_group().dispatch(
#                     hidden_states, router_logits, self.is_sequence_parallel)

#             # Matrix multiply.
#             final_hidden_states = self.quant_method.apply(
#                 layer=self,
#                 x=hidden_states,
#                 router_logits=router_logits,
#                 top_k=self.top_k,
#                 renormalize=self.renormalize,
#                 use_grouped_topk=self.use_grouped_topk,
#                 global_num_experts=self.global_num_experts,
#                 expert_map=self.expert_map,
#                 topk_group=self.topk_group,
#                 num_expert_group=self.num_expert_group,
#                 custom_routing_function=self.custom_routing_function,
#                 scoring_func=self.scoring_func,
#                 routed_scaling_factor=self.routed_scaling_factor,
#                 e_score_correction_bias=self.e_score_correction_bias,
#                 activation=self.activation,
#                 apply_router_weight_on_input=self.apply_router_weight_on_input,
#                 enable_eplb=self.enable_eplb,
#                 expert_load_view=self.expert_load_view,
#                 logical_to_physical_map=self.logical_to_physical_map,
#                 logical_replica_count=self.logical_replica_count,
#                 linear_weights=linear_weights
#             )

#             if shared_output is not None:
#                 assert not isinstance(final_hidden_states, tuple)
#                 assert self.shared_experts is not None
#                 final_hidden_states = (
#                     shared_output,
#                     final_hidden_states,
#                 )
#             elif self.zero_expert_num is not None and self.zero_expert_num > 0:
#                 assert isinstance(final_hidden_states, tuple)
#                 final_hidden_states, zero_expert_result = final_hidden_states

#             def reduce_output(states: torch.Tensor,
#                               do_combine: bool = True) -> torch.Tensor:
#                 if do_naive_dispatch_combine and do_combine:
#                     states = get_ep_group().combine(states,
#                                                     self.is_sequence_parallel)

#                 if (not self.is_sequence_parallel and self.reduce_results
#                         and (self.tp_size > 1 or self.ep_size > 1)):
#                     states = self.maybe_all_reduce_tensor_model_parallel(
#                         states)

#                 return states

#             if self.shared_experts is not None:
#                 return (
#                     reduce_output(final_hidden_states[0], do_combine=False),
#                     reduce_output(final_hidden_states[1]),
#                 )
#             elif self.zero_expert_num is not None and self.zero_expert_num > 0:
#                 assert isinstance(final_hidden_states, torch.Tensor)
#                 return reduce_output(final_hidden_states) + zero_expert_result
#             else:
#                 return reduce_output(final_hidden_states)

# FusedMoE=VllmFusedMoE











"""layer.py"""

from abc import abstractmethod
from collections.abc import Iterable
from contextlib import nullcontext
from enum import Enum
from typing import Callable, Literal, Optional, Union, get_args, overload

import torch
import torch.nn.functional as F
from torch.nn.parameter import UninitializedParameter

import vllm.envs as envs
from vllm.config import get_current_vllm_config
from vllm.config.parallel import ExpertPlacementStrategy
from vllm.distributed import (get_dp_group, get_ep_group,
                              get_tensor_model_parallel_world_size,
                              tensor_model_parallel_all_reduce)
from vllm.distributed.eplb.eplb_state import EplbState
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.custom_op import CustomOp
# yapf: disable
from vllm.model_executor.layers.fused_moe import FusedMoE as VllmFusedMoE
from vllm.model_executor.layers.fused_moe import FusedMoEMethodBase as VllmFusedMoEMethodBase
from vllm.model_executor.layers.fused_moe.layer import (
    UnquantizedFusedMoEMethod as VllmUnquantizedFusedMoEMethod)
from vllm.model_executor.layers.fused_moe.config import (
    FUSED_MOE_UNQUANTIZED_CONFIG, FusedMoEConfig, FusedMoEParallelConfig,
    FusedMoEQuantConfig, biased_moe_quant_config)
from vllm.model_executor.layers.fused_moe.fused_moe import (
    zero_experts_compute_triton)
# yapf: enable
from vllm.model_executor.layers.fused_moe.modular_kernel import (
    FusedMoEActivationFormat, FusedMoEModularKernel,
    FusedMoEPermuteExpertsUnpermute, FusedMoEPrepareAndFinalize)
from vllm.model_executor.layers.fused_moe.rocm_aiter_fused_moe import (
    is_rocm_aiter_moe_enabled)
from vllm.model_executor.layers.fused_moe.routing_simulator import (
    RoutingSimulator)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig, QuantizeMethodBase)
from vllm.model_executor.utils import set_weight_attrs
from vllm.platforms import current_platform
from vllm.platforms.interface import CpuArchEnum
from vllm.utils import (cdiv, direct_register_custom_op, has_deep_ep, has_pplx,
                        round_up)
from vllm.utils.flashinfer import has_flashinfer_cutlass_fused_moe
from vllm.v1.worker.ubatching import dbo_current_ubatch_id

class FusedMoEMethodBase(VllmFusedMoEMethodBase):
    """FusedMoEMethodBase"""
    moe: FusedMoEConfig

@CustomOp.register("vllm_kunlun_unquantized_fused_moe")
class UnquantizedFusedMoEMethod(VllmUnquantizedFusedMoEMethod):
    """UnquantizedFusedMoEMethod"""
    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool = False,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        global_num_experts: int = -1,
        expert_map: Optional[torch.Tensor] = None,
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = "softmax",
        e_score_correction_bias: Optional[torch.Tensor] = None,
        apply_router_weight_on_input: bool = False,
        activation: str = "silu",
        enable_eplb: bool = False,
        expert_load_view: Optional[torch.Tensor] = None,
        logical_to_physical_map: Optional[torch.Tensor] = None,
        logical_replica_count: Optional[torch.Tensor] = None,
        linear_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """apply"""
        if enable_eplb:
            raise NotImplementedError(
                "EPLB not supported for `UnquantizedFusedMoEMethod` yet.")

        return self.forward_kunlun(x=x,
                            layer=layer,
                            router_logits=router_logits,
                            top_k=top_k,
                            renormalize=renormalize,
                            use_grouped_topk=use_grouped_topk,
                            topk_group=topk_group,
                            num_expert_group=num_expert_group,
                            custom_routing_function=custom_routing_function,
                            linear_weights=linear_weights)

    def forward_kunlun(
            self,
            layer: torch.nn.Module,
            x: torch.Tensor,
            use_grouped_topk: bool,
            top_k: int,
            router_logits: torch.Tensor,
            linear_weights: torch.Tensor,
            renormalize: bool,
            topk_group: Optional[int] = None,
            num_expert_group: Optional[int] = None,
            custom_routing_function: Optional[Callable] = None
    ) -> torch.Tensor:
        """forward_kunlun"""
        from vllm_kunlun.ops._kunlun_ops import KunlunOps as ops
        if self.moe.use_ep:
            return ops.fused_moe_ep(x,
                             layer.w13_weight,
                             layer.w2_weight,
                             router_logits,
                             linear_weights,
                             self.moe.ep_rank,
                             top_k,
                             renormalize=renormalize,
                             inplace=True,
                             use_grouped_topk=use_grouped_topk,
                             num_expert_group=num_expert_group,
                             topk_group=topk_group
                             )
        else:
            return ops.fused_moe(x,
                             layer.w13_weight,
                             layer.w2_weight,
                             router_logits,
                             linear_weights,
                             top_k,
                             renormalize=renormalize,
                             inplace=True,
                             use_grouped_topk=use_grouped_topk,
                             num_expert_group=num_expert_group,
                             topk_group=topk_group
                             )

def maybe_roundup_hidden_size(
        hidden_size: int, act_dtype: torch.dtype,
        quant_config: Optional[QuantizationConfig],
        moe_parallel_config: FusedMoEParallelConfig) -> int:
    """
    Given layer hidden size and MoE configurations, round up hidden_size
    if necessary.
    
    Args:
        hidden_size: Layer hidden-size
        act_dtype: Data type of the layer activations.
        quant_config: Fused MoE quantization configuration.
        moe_parallel_config: Fused MoE parallelization strategy configuration.

    Return:
        Rounded up hidden_size if rounding up is required based on the configs.
        Original hidden size otherwise.
    """

    if (moe_parallel_config.use_deepep_ht_kernels):
        hidden_size = (
            DeepEPHTPrepareAndFinalize.maybe_roundup_layer_hidden_size(
                hidden_size, act_dtype))

    # we are padding globally so EP buffer allocation works
    if quant_config and quant_config.get_name() == "mxfp4":

        from vllm.model_executor.layers.quantization.mxfp4 import (
            Mxfp4Backend, get_mxfp4_backend)
        current_mxfp4_backend = get_mxfp4_backend()
        if (current_mxfp4_backend == Mxfp4Backend.SM90_FI_MXFP4_BF16
                or current_mxfp4_backend
                == Mxfp4Backend.SM100_FI_MXFP4_MXFP8_CUTLASS):
            hidden_size = round_up(hidden_size, 128)
        elif (current_platform.is_rocm() or current_mxfp4_backend
              == Mxfp4Backend.SM100_FI_MXFP4_MXFP8_TRTLLM
              or current_mxfp4_backend == Mxfp4Backend.SM100_FI_MXFP4_BF16):
            hidden_size = round_up(hidden_size, 256)

    return hidden_size

class FusedMoE(VllmFusedMoE):
    """FusedMoE layer for MoE models.

    This layer contains both MergedColumnParallel weights (gate_up_proj /
    w13) and RowParallelLinear weights (down_proj/ w2).

    Note: Mixtral uses w1, w2, and w3 for gate, up, and down_proj. We
    copy that naming convention here and handle any remapping in the
    load_weights function in each model implementation.

    Args:
        num_experts: Number of experts in the model
        top_k: Number of experts selected for each token
        hidden_size: Input hidden state size of the transformer
        intermediate_size: Intermediate size of the experts
        params_dtype: Data type for the parameters.
        reduce_results: Whether to all all_reduce on the output of the layer
        renormalize: Whether to renormalize the logits in the fused_moe kernel
        quant_config: Quantization configure.
        enable_eplb: Whether to enable expert parallelism load balancer.
    """

    def __init__(
        self,
        num_experts: int,  # Global number of experts
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        params_dtype: Optional[torch.dtype] = None,
        reduce_results: bool = False,
        renormalize: bool = True,
        use_grouped_topk: bool = False,
        num_expert_group: Optional[int] = 0,
        topk_group: Optional[int] = 0,
        quant_config: Optional[QuantizationConfig] = None,
        tp_size: Optional[int] = None,
        ep_size: Optional[int] = None,
        dp_size: Optional[int] = None,
        prefix: str = "",
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = "softmax",
        routed_scaling_factor: float = 1.0,
        e_score_correction_bias: Optional[torch.Tensor] = None,
        apply_router_weight_on_input: bool = False,
        activation: str = "silu",
        enable_eplb: bool = False,
        num_redundant_experts: int = 0,
        has_bias: bool = False,
        is_sequence_parallel=False,
        zero_expert_num: Optional[int] = 0,
        zero_expert_type: Optional[str] = None,
    ):
        super().__init__(
            num_experts=num_experts,  # Global number of experts
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            params_dtype=params_dtype,
            reduce_results=reduce_results,
            renormalize=renormalize,
            use_grouped_topk=use_grouped_topk,
            num_expert_group=num_expert_group,
            topk_group=topk_group,
            quant_config=quant_config,
            tp_size=tp_size,
            ep_size=ep_size,
            dp_size=dp_size,
            prefix=prefix,
            custom_routing_function=custom_routing_function,
            scoring_func=scoring_func,
            routed_scaling_factor=routed_scaling_factor,
            e_score_correction_bias=e_score_correction_bias,
            apply_router_weight_on_input=apply_router_weight_on_input,
            activation=activation,
            enable_eplb=enable_eplb,
            num_redundant_experts=num_redundant_experts,
            has_bias=has_bias,
            is_sequence_parallel=is_sequence_parallel,
            zero_expert_num=zero_expert_num,
            zero_expert_type=zero_expert_type)
        # if params_dtype is None:
        #     params_dtype = torch.get_default_dtype()
        # self.params_dtype = params_dtype

        # vllm_config = get_current_vllm_config()

        # # FIXME (varun): We should have a better way of inferring the activation
        # # datatype. This works for now as the tensor datatype entering the MoE
        # # operation is typically unquantized (i.e. float16/bfloat16).
        # if vllm_config.model_config is not None:
        #     moe_in_dtype = vllm_config.model_config.dtype
        # else:
        #     # TODO (bnell): This is a hack to get test_mixtral_moe to work
        #     # since model_config is not set in the pytest test.
        #     moe_in_dtype = params_dtype

        # tp_size_ = (tp_size if tp_size is not None else
        #             get_tensor_model_parallel_world_size())
        # dp_size_ = (dp_size
        #             if dp_size is not None else get_dp_group().world_size)

        # self.is_sequence_parallel = is_sequence_parallel
        # self.sp_size = tp_size_ if is_sequence_parallel else 1

        # self.moe_parallel_config: FusedMoEParallelConfig = (
        #     FusedMoEParallelConfig.make(
        #         tp_size_=tp_size_,
        #         dp_size_=dp_size_,
        #         vllm_parallel_config=vllm_config.parallel_config))

        # self.global_num_experts = num_experts + num_redundant_experts
        # self.zero_expert_num = zero_expert_num
        # self.zero_expert_type = zero_expert_type

        # # Round up hidden size if needed.
        # hidden_size = maybe_roundup_hidden_size(hidden_size, moe_in_dtype,
        #                                         quant_config,
        #                                         self.moe_parallel_config)

        # # For smuggling this layer into the fused moe custom op
        # compilation_config = vllm_config.compilation_config
        # if prefix in compilation_config.static_forward_context:
        #     raise ValueError("Duplicate layer name: {}".format(prefix))
        # compilation_config.static_forward_context[prefix] = self
        # self.layer_name = prefix

        # self.enable_eplb = enable_eplb
        # self.expert_load_view: Optional[torch.Tensor] = None
        # self.logical_to_physical_map: Optional[torch.Tensor] = None
        # self.logical_replica_count: Optional[torch.Tensor] = None

        # # Determine expert maps
        # if self.use_ep:
        #     if self.enable_eplb:
        #         assert self.global_num_experts % self.ep_size == 0, \
        #             "EPLB currently only supports even distribution of " \
        #             "experts across ranks."
        #     else:
        #         assert num_redundant_experts == 0, \
        #             "Redundant experts are only supported with EPLB."

        #     expert_placement_strategy = (
        #         vllm_config.parallel_config.expert_placement_strategy)
        #     if expert_placement_strategy == "round_robin":
        #         # TODO(Bruce): will support round robin expert placement with
        #         # EPLB enabled in the future.
        #         round_robin_supported = ((num_expert_group is not None
        #                                   and num_expert_group > 1)
        #                                  and num_redundant_experts == 0
        #                                  and not self.enable_eplb)

        #         if not round_robin_supported:
        #             logger.warning(
        #                 "Round-robin expert placement is only supported for "
        #                 "models with multiple expert groups and no redundant "
        #                 "experts. Falling back to linear expert placement.")
        #             expert_placement_strategy = "linear"

        #     self.expert_map: Optional[torch.Tensor]
        #     local_num_experts, expert_map = determine_expert_map(
        #         ep_size=self.ep_size,
        #         ep_rank=self.ep_rank,
        #         global_num_experts=self.global_num_experts,
        #         expert_placement_strategy=expert_placement_strategy,
        #     )
        #     self.local_num_experts = local_num_experts
        #     self.register_buffer("expert_map", expert_map)
        #     logger.info_once(
        #         "[EP Rank %s/%s] Expert parallelism is enabled. Expert "
        #         "placement strategy: %s. Local/global"
        #         " number of experts: %s/%s. Experts local to global index map:"
        #         " %s.", self.ep_rank, self.ep_size, expert_placement_strategy,
        #         self.local_num_experts, self.global_num_experts,
        #         get_compressed_expert_map(self.expert_map))
        # else:
        #     self.local_num_experts, self.expert_map = (self.global_num_experts,
        #                                                None)

        # self.top_k = top_k

        # assert intermediate_size % self.tp_size == 0
        # self.hidden_size = hidden_size
        # self.intermediate_size_per_partition = intermediate_size // self.tp_size
        # self.reduce_results = reduce_results
        # self.renormalize = renormalize
        # self.use_grouped_topk = use_grouped_topk
        # if self.use_grouped_topk:
        #     assert num_expert_group is not None and topk_group is not None
        # self.num_expert_group = num_expert_group
        # self.topk_group = topk_group
        # self.custom_routing_function = custom_routing_function
        # self.scoring_func = scoring_func
        # self.routed_scaling_factor = routed_scaling_factor
        # self.e_score_correction_bias = e_score_correction_bias
        # self.apply_router_weight_on_input = apply_router_weight_on_input
        # self.activation = activation

        # if self.scoring_func != "softmax" and not self.use_grouped_topk:
        #     raise ValueError("Only softmax scoring function is supported for "
        #                      "non-grouped topk.")

        # moe = FusedMoEConfig(
        #     num_experts=self.global_num_experts,
        #     experts_per_token=top_k,
        #     hidden_dim=hidden_size,
        #     num_local_experts=self.local_num_experts,
        #     moe_parallel_config=self.moe_parallel_config,
        #     in_dtype=moe_in_dtype,
        #     max_num_tokens=envs.VLLM_MOE_DP_CHUNK_SIZE,
        #     has_bias=has_bias,
        # )
        # self.moe_config = moe
        # self.moe_quant_config: Optional[FusedMoEQuantConfig] = None
        # self.quant_config = quant_config

        # # Note: get_quant_method will look at the layer's local_num_experts
        # # for heuristic purposes, so it must be initialized first.
        # quant_method: Optional[QuantizeMethodBase] = None
        # quant_method = (UnquantizedFusedMoEMethod(moe) if quant_config is None
        #                 else quant_config.get_quant_method(self, prefix))

        # assert quant_method is not None
        # assert isinstance(quant_method, FusedMoEMethodBase)
        # self.quant_method = quant_method

        # if self.enable_eplb:
        #     from vllm.model_executor.layers.quantization.fp8 import (
        #         Fp8MoEMethod)
        #     if not isinstance(quant_method,
        #                       (Fp8MoEMethod, UnquantizedFusedMoEMethod)):
        #         # TODO: Add support for additional quantization methods.
        #         # The implementation for other quantization methods does not
        #         # contain essential differences, but the current quant API
        #         # design causes duplicated work when extending to new
        #         # quantization methods, so I'm leaving it for now.
        #         # If you plan to add support for more quantization methods,
        #         # please refer to the implementation in `Fp8MoEMethod`.
        #         raise NotImplementedError("EPLB is only supported for FP8 "
        #                                   "quantization for now.")

        # moe_quant_params = {
        #     "num_experts": self.local_num_experts,
        #     "hidden_size": hidden_size,
        #     "intermediate_size_per_partition":
        #     self.intermediate_size_per_partition,
        #     "params_dtype": params_dtype,
        #     "weight_loader": self.weight_loader,
        # }
        # # need full intermediate size pre-sharding for WNA16 act order
        # if (self.quant_method.__class__.__name__
        #         in ("GPTQMarlinMoEMethod",
        #             "CompressedTensorsWNA16MarlinMoEMethod",
        #             "CompressedTensorsWNA16MoEMethod")):
        #     moe_quant_params["intermediate_size_full"] = intermediate_size

        # self.quant_method.create_weights(layer=self, **moe_quant_params)

        # # Chunked all2all staging tensor
        # self.batched_hidden_states: Optional[torch.Tensor] = None
        # self.batched_router_logits: Optional[torch.Tensor] = None

        # # TODO(bnell): flashinfer uses non-batched format.
        # # Does it really need a batched buffer?
        # if (self.moe_parallel_config.use_pplx_kernels
        #         or self.moe_parallel_config.use_deepep_ll_kernels
        #         or self.moe_config.use_flashinfer_cutlass_kernels):
        #     if vllm_config.parallel_config.enable_dbo:
        #         self.batched_hidden_states = torch.zeros(
        #             (2, moe.max_num_tokens, self.hidden_size),
        #             dtype=moe.in_dtype,
        #             device=torch.cuda.current_device())

        #         # Note here we use `num_experts` which is logical expert count
        #         self.batched_router_logits = torch.zeros(
        #             (2, moe.max_num_tokens, num_experts),
        #             dtype=moe.in_dtype,
        #             device=torch.cuda.current_device())
        #     else:
        #         self.batched_hidden_states = torch.zeros(
        #             (moe.max_num_tokens, self.hidden_size),
        #             dtype=moe.in_dtype,
        #             device=torch.cuda.current_device())

        #         # Note here we use `num_experts` which is logical expert count
        #         self.batched_router_logits = torch.zeros(
        #             (moe.max_num_tokens, num_experts),
        #             dtype=moe.in_dtype,
        #             device=torch.cuda.current_device())

    def forward_native(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        linear_weights: torch.Tensor = None
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        og_hidden_states = hidden_states.shape[-1]
        if self.hidden_size != og_hidden_states:
            hidden_states = F.pad(hidden_states,
                                  (0, self.hidden_size - og_hidden_states),
                                  mode='constant',
                                  value=0.0)

        if self.shared_experts is None:
            if current_platform.is_tpu():
                # TODO: Once the OOM issue for the TPU backend is resolved, we
                # will switch to using the moe_forward custom op.
                fused_output = self.forward_impl(hidden_states, router_logits)
                assert not isinstance(fused_output, tuple)
            else:
                fused_output = torch.ops.vllm.moe_forward(
                    hidden_states, router_logits, self.layer_name)
            return fused_output[..., :og_hidden_states]
        else:
            if current_platform.is_tpu():
                # TODO: Once the OOM issue for the TPU backend is resolved, we
                # will switch to using the moe_forward custom op.
                shared_output, fused_output = self.forward_impl(
                    hidden_states, router_logits)
            else:
                # shared_output, fused_output = torch.ops.vllm.moe_forward_shared(
                #     hidden_states, router_logits, self.layer_name)
                forward_context: ForwardContext = get_forward_context()
                self = forward_context.no_compile_layers[self.layer_name]
                assert self.shared_experts is not None
                shared_output, fused_output =  self.forward_impl(hidden_states, router_logits, linear_weights)
                # shared_output =  self.forward_impl(hidden_states, router_logits, linear_weights)
            return (shared_output[..., :og_hidden_states],
                    fused_output[..., :og_hidden_states])
            # return shared_output[..., :og_hidden_states]

    def forward(self, hidden_states: torch.Tensor,
                router_logits: torch.Tensor = None,
                linear_weights: torch.Tensor = None
        )-> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """forward"""
        # TODO: Once the OOM issue for the TPU backend is resolved, we will
        # switch to using the moe_forward custom op.
        # if current_platform.is_tpu():
        #     return self.forward_impl(hidden_states, router_logits)
        # else:
        #     forward_context: ForwardContext = get_forward_context()
        #     self = forward_context.no_compile_layers[self.layer_name]
        #     assert self.quant_method is not None
        #     return self.forward_impl(hidden_states, router_logits, linear_weights)
        #     # return torch.ops.vllm.moe_forward(hidden_states, router_logits,
        #     #                                   self.layer_name)

        return self.forward_native(hidden_states, router_logits, linear_weights)

    def forward_impl(self, hidden_states: torch.Tensor,
                     router_logits: torch.Tensor,
                     linear_weights: torch.Tensor = None
        )-> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """forward_impl"""
        assert self.quant_method is not None

        self.ensure_moe_quant_config()

        # Route to the chunked forward path using the FlashInfer Cutlass kernel
        # only when data parallelism (DP) is enabled.
        _use_flashinfer_cutlass_kernels = (self.dp_size > 1 and
                                           self.use_flashinfer_cutlass_kernels)

        if (self.moe_parallel_config.use_pplx_kernels
                or self.moe_parallel_config.use_deepep_ll_kernels
                or _use_flashinfer_cutlass_kernels):
            return self.forward_impl_chunked(hidden_states, router_logits)

        do_naive_dispatch_combine: bool = (
            self.dp_size > 1
            and not self.moe_parallel_config.use_deepep_ht_kernels
            and not self.moe_config.use_flashinfer_cutlass_kernels)

        # If there are shared experts but we are not using a modular kernel, the
        # shared experts must be called here
        if (not isinstance(self.quant_method.fused_experts,
                           FusedMoEModularKernel)
                and self.shared_experts is not None):
            shared_output = self.shared_experts(hidden_states)
        else:
            shared_output = None

        ctx = get_forward_context()
        sp_ctx = ctx.dp_metadata.sp_local_sizes(
            self.sp_size) if ctx.dp_metadata else nullcontext()

        with sp_ctx:
            if do_naive_dispatch_combine:
                hidden_states, router_logits = get_ep_group().dispatch(
                    hidden_states, router_logits, self.is_sequence_parallel)

            # Matrix multiply.
            final_hidden_states = self.quant_method.apply(
                layer=self,
                x=hidden_states,
                router_logits=router_logits,
                top_k=self.top_k,
                renormalize=self.renormalize,
                use_grouped_topk=self.use_grouped_topk,
                global_num_experts=self.global_num_experts,
                expert_map=self.expert_map,
                topk_group=self.topk_group,
                num_expert_group=self.num_expert_group,
                custom_routing_function=self.custom_routing_function,
                scoring_func=self.scoring_func,
                routed_scaling_factor=self.routed_scaling_factor,
                e_score_correction_bias=self.e_score_correction_bias,
                activation=self.activation,
                apply_router_weight_on_input=self.apply_router_weight_on_input,
                enable_eplb=self.enable_eplb,
                expert_load_view=self.expert_load_view,
                logical_to_physical_map=self.logical_to_physical_map,
                logical_replica_count=self.logical_replica_count,
                linear_weights=linear_weights
            )

            if shared_output is not None:
                assert not isinstance(final_hidden_states, tuple)
                assert self.shared_experts is not None
                final_hidden_states = (
                    shared_output,
                    final_hidden_states,
                )
            elif self.zero_expert_num is not None and self.zero_expert_num > 0:
                assert isinstance(final_hidden_states, tuple)
                final_hidden_states, zero_expert_result = final_hidden_states

            def reduce_output(states: torch.Tensor,
                              do_combine: bool = True) -> torch.Tensor:
                if do_naive_dispatch_combine and do_combine:
                    states = get_ep_group().combine(states,
                                                    self.is_sequence_parallel)

                if (not self.is_sequence_parallel and self.reduce_results
                        and (self.tp_size > 1 or self.ep_size > 1)):
                    states = self.maybe_all_reduce_tensor_model_parallel(
                        states)

                return states

            if self.shared_experts is not None:
                return (
                    reduce_output(final_hidden_states[0], do_combine=False),
                    reduce_output(final_hidden_states[1]),
                )
            elif self.zero_expert_num is not None and self.zero_expert_num > 0:
                assert isinstance(final_hidden_states, torch.Tensor)
                return reduce_output(final_hidden_states) + zero_expert_result
            else:
                return reduce_output(final_hidden_states)