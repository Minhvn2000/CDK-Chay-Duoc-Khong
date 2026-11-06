"""
Huấn luyện pipeline A1 cho DEPLOYMENT (Streamlit).
KHÔNG dùng để thay thế kết quả đánh giá A1 (train/test split).
Fit trên toàn bộ 751 hàng của dataset chính theo deployment policy đã chốt.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# Cấu hình
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = PROJECT_ROOT / "data" / "raw" / "llm-speed-runs-2026-10-22.6.csv"
ARTIFACT_PATH = PROJECT_ROOT / "models" / "cdk_a1_deployment.joblib"

TARGET = "decode_tps"
CAT_FEATURES = ["model", "hardware", "runtime", "workload", "quant"]
NUM_FEATURES = ["prompt_tokens", "batch_size"]
FEATURES = CAT_FEATURES + NUM_FEATURES


def build_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES),
            ("num", "passthrough", NUM_FEATURES),
        ],
        remainder="drop",
    )
    return Pipeline(
        steps=[
            ("pre", preprocessor),
            ("reg", LinearRegression()),
        ]
    )


def main() -> None:
    if not RAW_CSV.exists():
        raise FileNotFoundError(f"Không tìm thấy dataset: {RAW_CSV}")

    df = pd.read_csv(RAW_CSV)
    print(f"Loaded: {RAW_CSV}")
    print(f"Rows: {len(df)}, Columns: {len(df.columns)}")

    # Kiểm tra cột bắt buộc
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Thiếu cột: {missing}")

    # A1 preprocessing: quant missing -> Unknown
    data = df.copy()
    data["quant"] = data["quant"].fillna("Unknown")

    # Loại bỏ hàng thiếu target (đảm bảo an toàn; dataset hiện tại có đủ target)
    before = len(data)
    data = data.dropna(subset=[TARGET])
    dropped = before - len(data)
    if dropped:
        print(f"Cảnh báo: đã loại {dropped} hàng thiếu target (không mong đợi).")

    X = data[FEATURES].copy()
    y = data[TARGET].copy()

    # batch_size phải là 1 (theo dataset/policy)
    if not (X["batch_size"] == 1).all():
        print("Cảnh báo: batch_size có giá trị khác 1; deployment policy giữ batch_size=1.")

    pipe = build_pipeline()
    pipe.fit(X, y)

    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "artifact_type": "deployment",
            "model_name": "cdk_a1_deployment",
            "note": "Deployment pipeline (fit on full dataset). NOT for reporting A1 holdout metrics.",
            "target": TARGET,
            "features": FEATURES,
            "cat_features": CAT_FEATURES,
            "num_features": NUM_FEATURES,
            "rows_fit": int(len(X)),
            "pipeline": pipe,
        },
        ARTIFACT_PATH,
    )
    print(f"Saved artifact: {ARTIFACT_PATH}")
    print(f"Rows fit: {len(X)}")


if __name__ == "__main__":
    main()
