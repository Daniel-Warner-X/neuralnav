"""Unit tests for GPU recommender API routes."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from planner.api.app import create_app

client = TestClient(create_app())

ROUTE = "/api/v1/estimate"
MOCK_PATH = "planner.api.routes.gpu_recommender"

MOCK_PERFORMANCE_SUMMARY = {
    "estimated_best_performance": {
        "lowest_cost": {"gpu": "L40", "cost": 12.0},
        "highest_throughput": {"gpu": "H100", "throughput_tps": 900.0},
    },
    "gpu_results": {
        "H100": {
            "best_latency": {"throughput_tps": 900.0, "ttft_ms": 20.0, "itl_ms": 1.5},
            "cost": 25.0,
            "num_gpus": 1,
        },
        "L40": {
            "best_latency": {"throughput_tps": 450.0, "ttft_ms": 45.0, "itl_ms": 3.2},
            "cost": 12.0,
            "num_gpus": 1,
        },
    },
}


@pytest.mark.unit
@patch(f"{MOCK_PATH}.GPURecommender")
def test_estimate_success(mock_recommender_cls):
    mock_rec = MagicMock()
    mock_rec.get_gpu_results.return_value = (
        {"H100": MagicMock(), "L40": MagicMock()},
        {},  # no failed GPUs
    )
    mock_rec.get_performance_summary.return_value = MOCK_PERFORMANCE_SUMMARY
    mock_recommender_cls.return_value = mock_rec

    resp = client.post(
        ROUTE,
        json={
            "model_id": "Qwen/Qwen3-32B",
            "input_len": 512,
            "output_len": 128,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert "gpu_results" in data
    assert "estimated_best_performance" in data
    assert data["summary"]["failed_gpus"] == 0


@pytest.mark.unit
@patch(f"{MOCK_PATH}.GPURecommender")
def test_estimate_with_constraints(mock_recommender_cls):
    mock_rec = MagicMock()
    mock_rec.get_gpu_results.return_value = ({}, {"OldGPU": "does not fit"})
    mock_rec.get_performance_summary.return_value = {
        "estimated_best_performance": {},
        "gpu_results": {},
    }
    mock_recommender_cls.return_value = mock_rec

    resp = client.post(
        ROUTE,
        json={
            "model_id": "Qwen/Qwen3-32B",
            "input_len": 512,
            "output_len": 128,
            "max_ttft": 100.0,
            "max_itl": 10.0,
            "max_latency": 2.0,
            "gpu_list": ["H100", "OldGPU"],
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["failed_gpus"] == {"OldGPU": "does not fit"}
    assert data["input_parameters"]["max_ttft_ms"] == 100.0
    assert data["input_parameters"]["max_itl_ms"] == 10.0
    assert data["input_parameters"]["max_latency_s"] == 2.0


@pytest.mark.unit
def test_estimate_missing_required_fields():
    resp = client.post(ROUTE, json={"model_id": "Qwen/Qwen3-32B"})
    assert resp.status_code == 422  # missing input_len, output_len


@pytest.mark.unit
@patch(f"{MOCK_PATH}.GPURecommender", side_effect=Exception("gated repo"))
def test_estimate_hf_gated_error(mock_recommender_cls):
    resp = client.post(
        ROUTE,
        json={
            "model_id": "some/gated-model",
            "input_len": 512,
            "output_len": 128,
        },
    )
    assert resp.status_code == 403


@pytest.mark.unit
@patch(f"{MOCK_PATH}.GPURecommender", side_effect=Exception("connection refused"))
def test_estimate_generic_error(mock_recommender_cls):
    resp = client.post(
        ROUTE,
        json={
            "model_id": "some/model",
            "input_len": 512,
            "output_len": 128,
        },
    )
    assert resp.status_code == 500


@pytest.mark.unit
@patch(f"{MOCK_PATH}.GPU_SPECS", {"H100": {}, "A100": {}, "L40": {}})
@patch(f"{MOCK_PATH}.GPURecommender")
def test_estimate_default_gpu_list(mock_recommender_cls):
    """When gpu_list is omitted, all GPUs in GPU_SPECS are used."""
    mock_rec = MagicMock()
    mock_rec.get_gpu_results.return_value = ({}, {})
    mock_rec.get_performance_summary.return_value = {
        "estimated_best_performance": {},
        "gpu_results": {},
    }
    mock_recommender_cls.return_value = mock_rec

    resp = client.post(
        ROUTE,
        json={
            "model_id": "Qwen/Qwen3-32B",
            "input_len": 512,
            "output_len": 128,
        },
    )
    assert resp.status_code == 200
    # Verify all 3 GPUs from GPU_SPECS were passed to recommender
    call_kwargs = mock_recommender_cls.call_args[1]
    assert sorted(call_kwargs["gpu_list"]) == ["A100", "H100", "L40"]
