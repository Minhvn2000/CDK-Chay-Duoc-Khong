"""
CDK – Chạy Được Không
Streamlit MVP: dự đoán decode_tps bằng deployment pipeline A1.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Đảm bảo import được src khi chạy từ root project
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference import FEATURES, InferenceError, load_artifact, predict

RAW_CSV = PROJECT_ROOT / "data" / "raw" / "llm-speed-runs-2026-10-22.6.csv"
ARTIFACT_PATH = PROJECT_ROOT / "models" / "cdk_a1_deployment.joblib"

st.set_page_config(page_title="CDK – Chạy Được Không", page_icon="⚡", layout="centered")


@st.cache_data(show_spinner=False)
def load_options() -> dict:
    if not RAW_CSV.exists():
        raise FileNotFoundError(f"Không tìm thấy dữ liệu: {RAW_CSV}")
    df = pd.read_csv(RAW_CSV)
    required = ["model", "hardware", "runtime", "workload", "quant", "prompt_tokens"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset thiếu cột: {missing}")

    opts = {}
    for c in ["model", "hardware", "runtime", "workload"]:
        vals = sorted({str(x).strip() for x in df[c].dropna().astype(str) if str(x).strip()})
        opts[c] = vals
    quant_vals = sorted({str(x).strip() for x in df["quant"].dropna().astype(str) if str(x).strip()})
    opts["quant"] = ["Không rõ (Unknown)"] + [q for q in quant_vals if q != "Unknown"]
    opts["prompt_tokens_min"] = int(pd.to_numeric(df["prompt_tokens"], errors="coerce").min())
    opts["prompt_tokens_max"] = int(pd.to_numeric(df["prompt_tokens"], errors="coerce").max())
    return opts


@st.cache_resource(show_spinner=False)
def get_artifact():
    if not ARTIFACT_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy artifact: {ARTIFACT_PATH}")
    return load_artifact(ARTIFACT_PATH)


st.title("CDK – Chạy Được Không")
st.caption("Ước lượng tốc độ sinh token (`decode_tps`, token/s) từ cấu hình benchmark bằng mô hình Linear Regression (A1 deployment pipeline).")

try:
    options = load_options()
except Exception as e:
    st.error(f"Lỗi tải dữ liệu tùy chọn: {e}")
    st.stop()

try:
    artifact = get_artifact()
except Exception as e:
    st.error(f"Lỗi tải mô hình deployment: {e}")
    st.stop()

with st.form("predict_form"):
    st.subheader("Cấu hình đầu vào")
    col1, col2 = st.columns(2)
    with col1:
        model = st.selectbox("Model", options["model"], index=0)
        runtime = st.selectbox("Runtime", options["runtime"], index=0)
        workload = st.selectbox("Workload", options["workload"], index=0)
    with col2:
        hardware = st.selectbox("Hardware", options["hardware"], index=0)
        quant_label = st.selectbox("Quantization", options["quant"], index=0)
        prompt_tokens = st.number_input(
            "Prompt tokens",
            min_value=options["prompt_tokens_min"],
            max_value=options["prompt_tokens_max"],
            value=min(max(512, options["prompt_tokens_min"]), options["prompt_tokens_max"]),
            step=1,
        )

    st.info("`batch_size` cố định = **1** theo dữ liệu benchmark và deployment policy.")
    submitted = st.form_submit_button("Dự đoán decode_tps", type="primary")

if submitted:
    quant = "Unknown" if quant_label == "Không rõ (Unknown)" else quant_label
    payload = {
        "model": model,
        "hardware": hardware,
        "runtime": runtime,
        "workload": workload,
        "quant": quant,
        "prompt_tokens": int(prompt_tokens),
        "batch_size": 1,
    }
    try:
        pred = float(predict(payload, artifact)[0])
        st.subheader("Kết quả dự đoán")
        st.metric(label="decode_tps ước lượng", value=f"{pred:.2f} token/s")
        st.write("**Cấu hình đã sử dụng:**")
        st.json(payload)

        st.warning(
            "**Cảnh báo:** Kết quả là **ước lượng dựa trên benchmark dataset** (llm-speed-runs), chỉ mang tính tham khảo. "
            "Mô hình **chưa tổng quát hóa tốt** trên `model`/`hardware` chưa từng thấy trong dữ liệu. "
            "Kết quả **không phải** kết luận chắc chắn rằng một PC cụ thể “chạy được” hoặc “không chạy được” LLM."
        )
        st.caption(
            "Ghi chú: artifact deployment được fit trên toàn bộ dữ liệu để phục vụ demo; không hiển thị A1 holdout metrics như độ chính xác của deployment model."
        )
    except (InferenceError, FileNotFoundError, ValueError) as e:
        st.error(f"Không thể dự đoán: {e}")
    except Exception as e:
        st.error(f"Lỗi không mong đợi: {e}")
