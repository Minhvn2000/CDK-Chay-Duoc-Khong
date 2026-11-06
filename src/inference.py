"""
Inference module cho CDK A1 deployment pipeline.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd

TARGET = "decode_tps"
CAT_FEATURES = ["model", "hardware", "runtime", "workload", "quant"]
NUM_FEATURES = ["prompt_tokens", "batch_size"]
FEATURES = CAT_FEATURES + NUM_FEATURES


class InferenceError(ValueError):
    pass


def load_artifact(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy artifact: {path}")
    obj = joblib.load(path)
    if not isinstance(obj, dict) or "pipeline" not in obj:
        raise InferenceError("Artifact không hợp lệ: cần dict chứa key 'pipeline'.")
    expected = obj.get("features")
    if expected is not None and list(expected) != FEATURES:
        raise InferenceError(
            f"Artifact feature schema không khớp A1. Expected={FEATURES}, got={list(expected)}"
        )
    return obj


def _to_dataframe(payload: Mapping[str, Any] | pd.DataFrame) -> pd.DataFrame:
    if isinstance(payload, pd.DataFrame):
        df = payload.copy()
    elif isinstance(payload, Mapping):
        df = pd.DataFrame([dict(payload)])
    else:
        raise InferenceError("Input phải là dict/mapping hoặc pandas DataFrame.")

    # Kiểm tra schema
    missing = [c for c in FEATURES if c not in df.columns]
    extra = [c for c in df.columns if c not in FEATURES]
    if missing:
        raise InferenceError(f"Thiếu feature bắt buộc: {missing}")
    if extra:
        # Giữ nghiêm ngặt: chỉ cho phép đúng 7 cột để tránh nhầm lẫn
        df = df.drop(columns=extra)

    df = df[FEATURES].copy()

    # Xử lý quant missing -> Unknown (giống A1)
    df["quant"] = df["quant"].fillna("Unknown").astype(str).replace({"nan": "Unknown", "None": "Unknown", "": "Unknown"})

    # Categorical: ép string, giữ Unknown an toàn
    for c in CAT_FEATURES:
        df[c] = df[c].astype(str)

    # Numeric validation
    for c in NUM_FEATURES:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    bad_num = df[NUM_FEATURES].isna().any(axis=1)
    if bad_num.any():
        idx = df.index[bad_num].tolist()
        raise InferenceError(f"Giá trị numeric không hợp lệ tại dòng: {idx}. Cần prompt_tokens/batch_size là số.")

    # batch_size policy
    if not np.isclose(df["batch_size"], 1.0).all():
        raise InferenceError("batch_size phải bằng 1 theo deployment policy.")

    # prompt_tokens >= 0
    if (df["prompt_tokens"] < 0).any():
        raise InferenceError("prompt_tokens phải >= 0.")

    return df


def predict(payload: Mapping[str, Any] | pd.DataFrame, artifact: dict[str, Any]) -> np.ndarray:
    pipe = artifact["pipeline"]
    X = _to_dataframe(payload)
    pred = pipe.predict(X)
    pred = np.asarray(pred, dtype=float)
    if not np.isfinite(pred).all():
        raise InferenceError("Kết quả dự đoán không hữu hạn (NaN/Inf).")
    return pred


def predict_one(payload: Mapping[str, Any], artifact_path: str | Path) -> float:
    artifact = load_artifact(artifact_path)
    pred = predict(payload, artifact)
    return float(pred[0])
