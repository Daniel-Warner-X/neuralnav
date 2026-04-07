"""Capacity planner endpoints."""

import logging
import os
from typing import Any, NoReturn

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

import planner.capacity_planner as cp

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["capacity-planner"])


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------

def _get_hf_token() -> str | None:
    return os.getenv("HF_TOKEN") or None


def _handle_hf_error(e: Exception) -> NoReturn:
    """Raise the appropriate HTTPException for HuggingFace errors."""
    msg = str(e).lower()
    if "gated" in msg or "403" in msg or "unauthorized" in msg:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Model is gated. Set HF_TOKEN on the backend: {e}",
        )
    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail=f"Could not fetch model from HuggingFace: {e}",
    )


# ---------------------------------------------------------------------------
# /model-info schemas
# ---------------------------------------------------------------------------

class ModelInfoRequest(BaseModel):
    model_config = {"protected_namespaces": ()}
    model_id: str


class ModelInfoDetail(BaseModel):
    total_parameters: int
    parameters_by_dtype: dict[str, int]


class ArchitectureInfo(BaseModel):
    model_config = {"protected_namespaces": ()}
    architecture_name: str | None
    model_type: str
    num_hidden_layers: int
    num_attention_heads: int
    inference_dtype: str
    max_context_len: int
    is_moe: bool
    is_multimodal: bool
    num_experts: int | None = None


class QuantizationInfo(BaseModel):
    is_quantized: bool
    quant_method: str | None = None
    quant_bytes: float | None = None


class ActivationMemoryInfo(BaseModel):
    model_config = {"protected_namespaces": ()}
    activation_memory_gb: float
    source: str
    model_type: str
    validated_profiles: dict[str, float]
    base_constants: dict[str, float]


class MemoryBreakdownRow(BaseModel):
    dtype: str
    quantized_dtype: str
    bytes_per_param: float
    num_parameters: int
    memory_gb: float


class ModelInfoResponse(BaseModel):
    model_config = {"protected_namespaces": ()}
    success: bool
    model_id: str
    model_memory_gb: float
    possible_tp_values: list[int]
    model_info: ModelInfoDetail
    architecture: ArchitectureInfo
    quantization: QuantizationInfo
    activation_memory: ActivationMemoryInfo
    memory_breakdown: list[MemoryBreakdownRow]


# ---------------------------------------------------------------------------
# /model-info handler
# ---------------------------------------------------------------------------

@router.post("/model-info")
async def model_info(request: ModelInfoRequest) -> ModelInfoResponse:
    """Fetch model metadata from HuggingFace.

    Reads HF_TOKEN from backend environment — never from the request.
    """
    hf_token = _get_hf_token()

    try:
        model_config = cp.get_model_config_from_hf(request.model_id, hf_token)
    except Exception as e:
        _handle_hf_error(e)

    text_config = cp.get_text_config(model_config)

    # --- model_info ---
    try:
        params_by_dtype = cp.model_params_by_dtype(request.model_id, hf_token)
    except Exception:
        params_by_dtype = {}
    total_params = sum(params_by_dtype.values()) if params_by_dtype else cp.model_total_params(
        request.model_id, hf_token
    )

    memory_gb = cp.model_memory_req(request.model_id, model_config, hf_token)

    # --- architecture ---
    archs = getattr(model_config, "architectures", None) or []
    arch_name: str | None = archs[0] if archs else None
    is_moe_model = cp.is_moe(text_config)
    is_multimodal_model = cp.is_multimodal(model_config)
    if is_moe_model:
        model_type = "MoE"
    elif is_multimodal_model:
        model_type = "Multimodal"
    else:
        model_type = "Dense"

    # --- quantization ---
    is_quantized_model = cp.is_quantized(model_config)
    quant_method_val = cp.get_quant_method(model_config) if is_quantized_model else None
    quant_bytes_val = cp.get_quant_bytes(model_config) if is_quantized_model else None

    # --- activation memory ---
    if arch_name and arch_name in cp.VALIDATED_ACTIVATION_PROFILES:
        act_gb = cp.VALIDATED_ACTIVATION_PROFILES[arch_name]
        act_source = f"Validated profile for {arch_name}"
    elif is_moe_model:
        act_gb = cp.ACTIVATION_MEMORY_BASE_MOE_GIB
        act_source = "MoE default"
    elif is_multimodal_model:
        act_gb = cp.ACTIVATION_MEMORY_BASE_MULTIMODAL_GIB
        act_source = "Multimodal default"
    else:
        act_gb = cp.ACTIVATION_MEMORY_BASE_DENSE_GIB
        act_source = "Dense default"

    # --- memory breakdown (one row per dtype) ---
    breakdown: list[MemoryBreakdownRow] = []
    quant_method_str = quant_method_val or ""
    quant_bytes_float = quant_bytes_val or 0.0
    for dtype, param_count in params_by_dtype.items():
        try:
            param_bytes = cp.precision_to_byte(dtype)
        except ValueError:
            param_bytes = 0.0
        if param_bytes >= 2 or not quant_method_str:
            q_dtype = dtype
            q_bytes = param_bytes
            mem_gb = cp.parameter_memory_req(param_count, dtype) if param_bytes > 0 else 0.0
        else:
            q_dtype = quant_method_str
            q_bytes = quant_bytes_float
            mem_gb = cp.parameter_precision_memory_req(param_count, quant_bytes_float)
        breakdown.append(MemoryBreakdownRow(
            dtype=dtype,
            quantized_dtype=q_dtype,
            bytes_per_param=q_bytes,
            num_parameters=param_count,
            memory_gb=round(mem_gb, 2),
        ))

    return ModelInfoResponse(
        success=True,
        model_id=request.model_id,
        model_memory_gb=round(memory_gb, 2),
        possible_tp_values=cp.find_possible_tp(model_config),
        model_info=ModelInfoDetail(
            total_parameters=total_params,
            parameters_by_dtype=params_by_dtype,
        ),
        architecture=ArchitectureInfo(
            architecture_name=arch_name,
            model_type=model_type,
            num_hidden_layers=text_config.num_hidden_layers,
            num_attention_heads=text_config.num_attention_heads,
            inference_dtype=cp.inference_dtype(model_config),
            max_context_len=cp.max_context_len(text_config),
            is_moe=is_moe_model,
            is_multimodal=is_multimodal_model,
            num_experts=cp.get_num_experts(model_config) if is_moe_model else None,
        ),
        quantization=QuantizationInfo(
            is_quantized=is_quantized_model,
            quant_method=quant_method_val,
            quant_bytes=quant_bytes_val,
        ),
        activation_memory=ActivationMemoryInfo(
            activation_memory_gb=act_gb,
            source=act_source,
            model_type=model_type,
            validated_profiles=dict(cp.VALIDATED_ACTIVATION_PROFILES),
            base_constants={
                "dense_gib": cp.ACTIVATION_MEMORY_BASE_DENSE_GIB,
                "moe_gib": cp.ACTIVATION_MEMORY_BASE_MOE_GIB,
                "multimodal_gib": cp.ACTIVATION_MEMORY_BASE_MULTIMODAL_GIB,
            },
        ),
        memory_breakdown=breakdown,
    )


# ---------------------------------------------------------------------------
# /calculate schemas
# ---------------------------------------------------------------------------

class CalculateRequest(BaseModel):
    model_config = {"protected_namespaces": ()}
    model_id: str
    max_model_len: int | None = None
    batch_size: int = 1
    gpu_memory: float | None = None
    tp: int = 1
    pp: int = 1
    dp: int = 1
    gpu_mem_util: float = 0.9
    block_size: int = 16


class KVCacheDetailSchema(BaseModel):
    attention_type: str
    kv_data_type: str
    precision_in_bytes: float
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    num_attention_group: int
    head_dimension: int
    per_token_memory_bytes: int
    per_request_kv_cache_bytes: int
    per_request_kv_cache_gb: float
    kv_cache_size_gb: float
    context_len: int
    batch_size: int
    kv_lora_rank: int | None = None
    qk_rope_head_dim: int | None = None


class CalculateResponse(BaseModel):
    model_config = {"protected_namespaces": ()}
    success: bool
    input_parameters: dict[str, Any]
    kv_cache_detail: KVCacheDetailSchema
    warnings: list[str]
    per_gpu_model_memory_gb: float | None = None
    total_gpus_required: int | None = None
    allocatable_kv_cache_memory_gb: float | None = None
    max_concurrent_requests: int | None = None
    total_kv_cache_blocks: int | None = None
    activation_memory_gb: float | None = None
    cuda_graph_memory_gb: float | None = None
    non_torch_memory_gb: float | None = None
    model_memory_gb: float | None = None
    available_gpu_memory_gb: float | None = None


# ---------------------------------------------------------------------------
# /calculate handler
# ---------------------------------------------------------------------------

@router.post("/calculate")
async def calculate(request: CalculateRequest) -> CalculateResponse:
    """Run capacity planning calculations for a given model and hardware config."""
    hf_token = _get_hf_token()
    warnings_list: list[str] = []

    try:
        model_config = cp.get_model_config_from_hf(request.model_id, hf_token)
    except Exception as e:
        _handle_hf_error(e)

    text_config = cp.get_text_config(model_config)

    # Resolve max_model_len
    max_model_len_auto = False
    if request.max_model_len == -1:
        if request.gpu_memory is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="max_model_len=-1 requires gpu_memory to be specified for auto-calculation",
            )
        max_len = cp.auto_max_model_len(
            request.model_id, model_config,
            gpu_memory=int(request.gpu_memory),
            gpu_mem_util=request.gpu_mem_util,
            tp=request.tp, pp=request.pp, dp=request.dp,
            hf_token=hf_token,
        )
        if max_len == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Model does not fit in available GPU memory. Increase gpu_memory, tp, or pp.",
            )
        if max_len < 128:
            warnings_list.append(
                f"Auto-calculated max_model_len is {max_len} tokens, which may be too small for practical use."
            )
        max_model_len_auto = True
    elif request.max_model_len is not None:
        max_len = request.max_model_len
    else:
        max_len = cp.max_context_len(text_config)

    # Validate TP
    possible_tp = cp.find_possible_tp(model_config)
    if request.tp not in possible_tp:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid tp value {request.tp}. Valid values for this model: {possible_tp}",
        )

    # KV cache detail
    kv = cp.KVCacheDetail(request.model_id, model_config, max_len, request.batch_size)

    input_params: dict[str, Any] = {
        "model": request.model_id,
        "max_model_len": max_len,
        "batch_size": request.batch_size,
    }
    if max_model_len_auto:
        input_params["max_model_len_auto"] = True

    response = CalculateResponse(
        success=True,
        input_parameters=input_params,
        kv_cache_detail=KVCacheDetailSchema(
            attention_type=str(kv.attention_type),
            kv_data_type=kv.kv_data_type,
            precision_in_bytes=kv.precision_in_bytes,
            num_hidden_layers=kv.num_hidden_layers,
            num_attention_heads=kv.num_attention_heads,
            num_key_value_heads=kv.num_key_value_heads,
            num_attention_group=kv.num_attention_group,
            head_dimension=kv.head_dimension,
            per_token_memory_bytes=kv.per_token_memory_bytes,
            per_request_kv_cache_bytes=kv.per_request_kv_cache_bytes,
            per_request_kv_cache_gb=round(kv.per_request_kv_cache_gb, 4),
            kv_cache_size_gb=round(kv.kv_cache_size_gb, 2),
            context_len=kv.context_len,
            batch_size=kv.batch_size,
            kv_lora_rank=kv.kv_lora_rank,
            qk_rope_head_dim=kv.qk_rope_head_dim,
        ),
        warnings=warnings_list,
    )

    if request.gpu_memory is not None:
        gpu_memory_int = int(request.gpu_memory)
        input_params.update({
            "tp": request.tp, "pp": request.pp, "dp": request.dp,
            "gpu_mem_util": request.gpu_mem_util, "block_size": request.block_size,
        })
        response.per_gpu_model_memory_gb = round(
            cp.per_gpu_model_memory_required(request.model_id, model_config, request.tp, request.pp, hf_token), 2
        )
        response.total_gpus_required = cp.gpus_required(request.tp, request.pp, request.dp)
        response.allocatable_kv_cache_memory_gb = round(
            cp.allocatable_kv_cache_memory(
                request.model_id, model_config, gpu_memory_int, request.gpu_mem_util,
                request.tp, request.pp, request.dp,
                max_model_len=max_len, batch_size=request.batch_size, hf_token=hf_token,
            ), 2
        )
        response.max_concurrent_requests = cp.max_concurrent_requests(
            request.model_id, model_config, max_len, gpu_memory_int,
            request.gpu_mem_util, batch_size=request.batch_size,
            tp=request.tp, pp=request.pp, dp=request.dp, hf_token=hf_token,
        )
        response.total_kv_cache_blocks = int(cp.total_kv_cache_blocks(
            request.model_id, model_config, max_len, gpu_memory_int,
            request.gpu_mem_util, request.batch_size, request.block_size,
            request.tp, request.pp, request.dp, hf_token=hf_token,
        ))
        response.activation_memory_gb = round(cp.estimate_vllm_activation_memory(model_config, tp=request.tp), 4)
        response.cuda_graph_memory_gb = round(cp.estimate_vllm_cuda_graph_memory(), 4)
        response.non_torch_memory_gb = round(cp.estimate_vllm_non_torch_memory(request.tp), 4)
        response.model_memory_gb = round(cp.model_memory_req(request.model_id, model_config, hf_token), 2)
        response.available_gpu_memory_gb = round(cp.available_gpu_memory(gpu_memory_int, request.gpu_mem_util), 2)

    return response
