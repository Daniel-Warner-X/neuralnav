"""Unit tests for capacity planner API routes."""

import os
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from planner.api.app import create_app

client = TestClient(create_app())

# Minimal mock AutoConfig with the attributes the route accesses
def _mock_model_config(
    arch="LlamaForCausalLM",
    num_hidden_layers=32,
    num_attention_heads=32,
    num_key_value_heads=8,
    hidden_size=4096,
    is_moe=False,
    is_multimodal=False,
    is_quantized=False,
):
    cfg = MagicMock()
    cfg.architectures = [arch]
    cfg.num_hidden_layers = num_hidden_layers
    cfg.num_attention_heads = num_attention_heads
    cfg.num_key_value_heads = num_key_value_heads
    cfg.hidden_size = hidden_size
    # text_config is the same object for non-multimodal
    cfg.text_config = cfg
    if not is_quantized:
        del cfg.quantization_config
        type(cfg).quantization_config = property(
            lambda self: (_ for _ in ()).throw(AttributeError("no quant"))
        )
    if not is_moe:
        # Remove MoE indicator attributes so is_moe() returns False
        for attr in ['n_routed_experts', 'n_shared_experts', 'num_experts', 'num_experts_per_tok']:
            if hasattr(cfg, attr):
                delattr(cfg, attr)
            # Capture attr in the closure
            def _raise_attr_error(a: str = attr) -> property:
                return property(lambda self: (_ for _ in ()).throw(AttributeError(f"no {a}")))
            setattr(type(cfg), attr, _raise_attr_error())
    return cfg


ROUTE = "/api/v1/model-info"

MOCK_PATH = "planner.capacity_planner"


@pytest.mark.unit
@patch(f"{MOCK_PATH}.model_memory_req", return_value=14.0)
@patch(f"{MOCK_PATH}.model_total_params", return_value=7_000_000_000)
@patch(f"{MOCK_PATH}.model_params_by_dtype", return_value={"BF16": 7_000_000_000})
@patch(f"{MOCK_PATH}.find_possible_tp", return_value=[1, 2, 4])
@patch(f"{MOCK_PATH}.get_model_config_from_hf")
def test_model_info_success(
    mock_config, mock_tp, mock_params, mock_total, mock_mem
):
    mock_config.return_value = _mock_model_config()
    resp = client.post(ROUTE, json={"model_id": "meta-llama/Llama-3-8B"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert data["model_memory_gb"] == 14.0
    assert data["possible_tp_values"] == [1, 2, 4]
    assert data["model_info"]["total_parameters"] == 7_000_000_000
    assert data["model_info"]["parameters_by_dtype"] == {"BF16": 7_000_000_000}
    assert data["architecture"]["architecture_name"] == "LlamaForCausalLM"
    assert data["architecture"]["is_moe"] is False
    assert data["architecture"]["model_type"] == "Dense"
    assert data["quantization"]["is_quantized"] is False
    assert "activation_memory_gb" in data["activation_memory"]
    assert "validated_profiles" in data["activation_memory"]
    assert "base_constants" in data["activation_memory"]
    assert isinstance(data["memory_breakdown"], list)


@pytest.mark.unit
@patch(f"{MOCK_PATH}.get_model_config_from_hf", side_effect=Exception("gated repo"))
def test_model_info_gated_model(mock_config):
    resp = client.post(ROUTE, json={"model_id": "meta-llama/Llama-3-70B"})
    assert resp.status_code == 403


@pytest.mark.unit
@patch(f"{MOCK_PATH}.get_model_config_from_hf", side_effect=Exception("repo not found"))
def test_model_info_not_found(mock_config):
    resp = client.post(ROUTE, json={"model_id": "nonexistent/model"})
    assert resp.status_code == 400


@pytest.mark.unit
def test_model_info_missing_model_id():
    resp = client.post(ROUTE, json={})
    assert resp.status_code == 422  # Pydantic validation error


# ---------------------------------------------------------------------------
# /calculate tests
# ---------------------------------------------------------------------------

CALC_ROUTE = "/api/v1/calculate"


def _mock_kv_detail(
    attention_type="Grouped-query attention",
    kv_data_type="fp8",
    precision_in_bytes=1,
    num_hidden_layers=32,
    num_attention_heads=32,
    num_key_value_heads=8,
    head_dimension=128,
    per_token_memory_bytes=65536,
    per_request_kv_cache_bytes=2147483648,
    per_request_kv_cache_gb=2.0,
    kv_cache_size_gb=2.0,
    context_len=4096,
    batch_size=1,
    kv_lora_rank=None,
    qk_rope_head_dim=None,
):
    kv = MagicMock()
    kv.attention_type = attention_type
    kv.kv_data_type = kv_data_type
    kv.precision_in_bytes = precision_in_bytes
    kv.num_hidden_layers = num_hidden_layers
    kv.num_attention_heads = num_attention_heads
    kv.num_key_value_heads = num_key_value_heads
    kv.num_attention_group = num_attention_heads // num_key_value_heads
    kv.head_dimension = head_dimension
    kv.per_token_memory_bytes = per_token_memory_bytes
    kv.per_request_kv_cache_bytes = per_request_kv_cache_bytes
    kv.per_request_kv_cache_gb = per_request_kv_cache_gb
    kv.kv_cache_size_gb = kv_cache_size_gb
    kv.context_len = context_len
    kv.batch_size = batch_size
    kv.kv_lora_rank = kv_lora_rank
    kv.qk_rope_head_dim = qk_rope_head_dim
    return kv


@pytest.mark.unit
@patch(f"{MOCK_PATH}.KVCacheDetail", return_value=_mock_kv_detail())
@patch(f"{MOCK_PATH}.find_possible_tp", return_value=[1, 2, 4])
@patch(f"{MOCK_PATH}.max_context_len", return_value=32768)
@patch(f"{MOCK_PATH}.get_text_config", side_effect=lambda cfg: cfg)
@patch(f"{MOCK_PATH}.get_model_config_from_hf")
def test_calculate_basic_no_gpu(
    mock_config, mock_text, mock_ctx, mock_tp, mock_kv
):
    mock_config.return_value = _mock_model_config()
    resp = client.post(CALC_ROUTE, json={"model_id": "meta-llama/Llama-3-8B"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert "kv_cache_detail" in data
    assert data["kv_cache_detail"]["precision_in_bytes"] == 1
    assert data["kv_cache_detail"]["num_attention_group"] == 4
    assert "per_request_kv_cache_bytes" in data["kv_cache_detail"]
    assert data["per_gpu_model_memory_gb"] is None
    assert data["warnings"] == []


@pytest.mark.unit
@patch(f"{MOCK_PATH}.available_gpu_memory", return_value=72.0)
@patch(f"{MOCK_PATH}.model_memory_req", return_value=14.0)
@patch(f"{MOCK_PATH}.estimate_vllm_non_torch_memory", return_value=0.15)
@patch(f"{MOCK_PATH}.estimate_vllm_cuda_graph_memory", return_value=0.5)
@patch(f"{MOCK_PATH}.estimate_vllm_activation_memory", return_value=5.5)
@patch(f"{MOCK_PATH}.total_kv_cache_blocks", return_value=512)
@patch(f"{MOCK_PATH}.max_concurrent_requests", return_value=4)
@patch(f"{MOCK_PATH}.allocatable_kv_cache_memory", return_value=10.5)
@patch(f"{MOCK_PATH}.gpus_required", return_value=1)
@patch(f"{MOCK_PATH}.per_gpu_model_memory_required", return_value=14.0)
@patch(f"{MOCK_PATH}.KVCacheDetail", return_value=_mock_kv_detail())
@patch(f"{MOCK_PATH}.find_possible_tp", return_value=[1, 2, 4])
@patch(f"{MOCK_PATH}.max_context_len", return_value=32768)
@patch(f"{MOCK_PATH}.get_text_config", side_effect=lambda cfg: cfg)
@patch(f"{MOCK_PATH}.get_model_config_from_hf")
def test_calculate_with_gpu_memory(
    mock_config, mock_text, mock_ctx, mock_tp, mock_kv,
    mock_per_gpu, mock_gpus, mock_alloc, mock_conc, mock_blocks,
    mock_act, mock_cuda, mock_non_torch, mock_model_mem, mock_avail
):
    mock_config.return_value = _mock_model_config()
    resp = client.post(CALC_ROUTE, json={
        "model_id": "meta-llama/Llama-3-8B",
        "gpu_memory": 80,
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["per_gpu_model_memory_gb"] == 14.0
    assert data["total_gpus_required"] == 1
    assert data["allocatable_kv_cache_memory_gb"] == 10.5
    assert data["max_concurrent_requests"] == 4
    assert data["total_kv_cache_blocks"] == 512
    assert data["activation_memory_gb"] == 5.5
    assert data["cuda_graph_memory_gb"] == 0.5
    assert data["non_torch_memory_gb"] == 0.15
    assert data["model_memory_gb"] == 14.0
    assert data["available_gpu_memory_gb"] == 72.0


@pytest.mark.unit
@patch(f"{MOCK_PATH}.get_model_config_from_hf")
def test_calculate_auto_max_model_len_requires_gpu_memory(mock_config):
    mock_config.return_value = _mock_model_config()
    resp = client.post(CALC_ROUTE, json={
        "model_id": "meta-llama/Llama-3-8B",
        "max_model_len": -1,
        # no gpu_memory
    })
    assert resp.status_code == 400
    assert "gpu_memory" in resp.json()["detail"]


@pytest.mark.unit
@patch(f"{MOCK_PATH}.find_possible_tp", return_value=[1, 2, 4])
@patch(f"{MOCK_PATH}.max_context_len", return_value=32768)
@patch(f"{MOCK_PATH}.get_text_config", side_effect=lambda cfg: cfg)
@patch(f"{MOCK_PATH}.get_model_config_from_hf")
def test_calculate_invalid_tp(mock_config, mock_text, mock_ctx, mock_tp):
    mock_config.return_value = _mock_model_config()
    resp = client.post(CALC_ROUTE, json={
        "model_id": "meta-llama/Llama-3-8B",
        "tp": 3,  # invalid: not in [1, 2, 4]
    })
    assert resp.status_code == 400
    assert "tp" in resp.json()["detail"].lower()


@pytest.mark.unit
@patch(f"{MOCK_PATH}.available_gpu_memory", return_value=72.0)
@patch(f"{MOCK_PATH}.model_memory_req", return_value=14.0)
@patch(f"{MOCK_PATH}.estimate_vllm_non_torch_memory", return_value=0.15)
@patch(f"{MOCK_PATH}.estimate_vllm_cuda_graph_memory", return_value=0.5)
@patch(f"{MOCK_PATH}.estimate_vllm_activation_memory", return_value=5.5)
@patch(f"{MOCK_PATH}.total_kv_cache_blocks", return_value=512)
@patch(f"{MOCK_PATH}.max_concurrent_requests", return_value=4)
@patch(f"{MOCK_PATH}.allocatable_kv_cache_memory", return_value=10.5)
@patch(f"{MOCK_PATH}.gpus_required", return_value=1)
@patch(f"{MOCK_PATH}.per_gpu_model_memory_required", return_value=14.0)
@patch(f"{MOCK_PATH}.auto_max_model_len", return_value=64)  # < 128 → warning
@patch(f"{MOCK_PATH}.KVCacheDetail", return_value=_mock_kv_detail())
@patch(f"{MOCK_PATH}.find_possible_tp", return_value=[1])
@patch(f"{MOCK_PATH}.get_text_config", side_effect=lambda cfg: cfg)
@patch(f"{MOCK_PATH}.get_model_config_from_hf")
def test_calculate_auto_max_model_len_small_warning(
    mock_config, mock_text, mock_tp, mock_kv, mock_auto,
    mock_per_gpu, mock_gpus, mock_alloc, mock_conc, mock_blocks,
    mock_act, mock_cuda, mock_non_torch, mock_model_mem, mock_avail
):
    mock_config.return_value = _mock_model_config()
    resp = client.post(CALC_ROUTE, json={
        "model_id": "meta-llama/Llama-3-8B",
        "max_model_len": -1,
        "gpu_memory": 80,
    })
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["warnings"]) > 0
    assert "64" in data["warnings"][0]
