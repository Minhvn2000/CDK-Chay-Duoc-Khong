"""Regression tests cho CDK deployment guard trong `app.py`.

Chỉ kiểm thử UI/guard; không retrain, không sửa artifact/dataset/inference.
Chạy: python -m pytest tests/test_app_guard.py -q
"""
from __future__ import annotations

import math

import pytest
from streamlit.testing.v1 import AppTest

import src.inference as inference

APP = "app.py"

# A/H: tổ hợp có trong dataset, prompt nằm đúng khoảng đã quan sát.
SUPPORTED = {
    "model": "Qwen3.6-27B-Q4_K_M.gguf",
    "hardware": "RTX 5090 (32GB)",
    "runtime": "llama.cpp",
    "workload": "chat-short",
    "quant": "Không rõ (Unknown)",
    "prompt_tokens": 0,
}

# B: đủ 5 category nhưng tổ hợp chưa từng xuất hiện.
UNSUPPORTED = {
    "model": "Codestral-22B-v0.1",
    "hardware": "2x RTX 3090 (24GB)",
    "runtime": "llama.cpp",
    "workload": "agent-trace",
    "quant": "4bit",
    "prompt_tokens": 0,
}

# C: tổ hợp có thật, nhưng prompt thấp hơn min quan sát (min = 3362).
BELOW_MIN = {
    "model": "Codestral-22B-v0.1",
    "hardware": "RTX 5090 (32GB)",
    "runtime": "llama.cpp",
    "workload": "agent-trace",
    "quant": "Không rõ (Unknown)",
    "prompt_tokens": 0,
}

# D: tổ hợp có thật, nhưng prompt cao hơn max quan sát (max = 0).
ABOVE_MAX = {
    "model": "Codestral-22B-v0.1",
    "hardware": "RTX 5090 (32GB)",
    "runtime": "llama.cpp",
    "workload": "chat-long",
    "quant": "Không rõ (Unknown)",
    "prompt_tokens": 1,
}

# G: dự đoán thật nằm trong (0, 5) TPS.
LOW_POSITIVE = {
    "model": "Flint-40B-A2.5B-UD-cpumoe12-ctx8k",
    "hardware": "RTX A4000 (16GB) + AMD EPYC 7453 28-Core Processor (56c) + 504GB",
    "runtime": "llama.cpp",
    "workload": "chat-long",
    "quant": "Q3_K_XL",
    "prompt_tokens": 3014,
}


def _run_app(config: dict, monkeypatch: pytest.MonkeyPatch | None = None, fake_predict=None) -> AppTest:
    if fake_predict is not None:
        assert monkeypatch is not None
        monkeypatch.setattr(inference, "predict", fake_predict)

    at = AppTest.from_file(APP, default_timeout=60)
    at.run()

    at.selectbox[0].set_value(config["model"])
    at.selectbox[1].set_value(config["runtime"])
    at.selectbox[2].set_value(config["workload"])
    at.selectbox[3].set_value(config["hardware"])
    at.selectbox[4].set_value(config["quant"])
    at.number_input[0].set_value(config["prompt_tokens"])
    at.button[0].click()
    at.run()
    return at


def _metric_value(at: AppTest) -> float:
    assert len(at.metric) == 1, f"expected one metric, got {[m.value for m in at.metric]}"
    value = at.metric[0].value
    assert value.endswith(" token/s")
    return float(value.split()[0])


def test_a_supported_configuration_predicts(monkeypatch):
    at = _run_app(SUPPORTED)
    assert not at.exception
    assert not at.error
    assert _metric_value(at) >= 5
    assert any("ước lượng dựa trên benchmark dataset" in w.value for w in at.warning)


def test_b_unsupported_configuration_blocks_without_inference(monkeypatch):
    calls = []

    def must_not_call(payload, artifact):
        calls.append(payload)
        raise AssertionError("predict() phải không được gọi cho cấu hình không hỗ trợ")

    at = _run_app(UNSUPPORTED, monkeypatch, must_not_call)

    assert calls == []
    assert not at.exception
    assert not at.metric
    assert any("chưa từng xuất hiện" in e.value for e in at.error)


def test_c_supported_configuration_below_observed_min_blocks(monkeypatch):
    calls = []

    def must_not_call(payload, artifact):
        calls.append(payload)
        raise AssertionError("predict() phải không được gọi khi prompt dưới min")

    at = _run_app(BELOW_MIN, monkeypatch, must_not_call)

    assert calls == []
    assert not at.metric
    assert any("thấp hơn" in e.value and "3362" in e.value for e in at.error)


def test_d_supported_configuration_above_observed_max_blocks(monkeypatch):
    calls = []

    def must_not_call(payload, artifact):
        calls.append(payload)
        raise AssertionError("predict() phải không được gọi khi prompt trên max")

    at = _run_app(ABOVE_MAX, monkeypatch, must_not_call)

    assert calls == []
    assert not at.metric
    assert any("cao hơn" in e.value and "0" in e.value for e in at.error)


def test_e_non_positive_prediction_is_not_shown_as_tps(monkeypatch):
    calls = []

    def negative(payload, artifact):
        calls.append(payload)
        return [-5.0]

    at = _run_app(SUPPORTED, monkeypatch, negative)

    assert len(calls) == 1
    assert not at.exception
    assert not at.metric
    assert any("<= 0" in e.value and "KHÔNG phải TPS hợp lệ" in e.value for e in at.error)


@pytest.mark.parametrize("bad_value", [math.nan, math.inf, -math.inf])
def test_f_non_finite_prediction_is_rejected(monkeypatch, bad_value):
    def non_finite(payload, artifact):
        return [bad_value]

    at = _run_app(SUPPORTED, monkeypatch, non_finite)

    assert not at.exception
    assert not at.metric
    assert any("không hữu hạn" in e.value for e in at.error)


def test_g_low_positive_prediction_is_shown_with_strong_warning(monkeypatch):
    at = _run_app(LOW_POSITIVE)

    assert not at.exception
    assert not at.error
    assert 0 < _metric_value(at) < 5
    assert any("Cảnh báo mạnh" in w.value and "< 5 token/s" in w.value for w in at.warning)


def test_h_normal_positive_prediction_keeps_disclaimer(monkeypatch):
    at = _run_app(SUPPORTED)

    assert not at.exception
    assert not at.error
    assert _metric_value(at) >= 5
    assert any("ước lượng dựa trên benchmark dataset" in w.value for w in at.warning)
    assert any("chưa tổng quát hóa tốt" in w.value for w in at.warning)
    assert not any("Cảnh báo mạnh" in w.value for w in at.warning)
