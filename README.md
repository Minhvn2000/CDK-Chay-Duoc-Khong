# CDK – Chạy Được Không

Dự án phân tích và dự đoán tốc độ sinh token (`decode_tps`, tokens/s) của LLM chạy local bằng **Linear Regression**, phục vụ báo cáo cuối khóa MindX với data storytelling.

## 1. Research Questions

- Các yếu tố nào ảnh hưởng đến `decode_tps` trong tập dữ liệu benchmark hiện tại?
- Có thể dự đoán `decode_tps` từ cấu hình benchmark (`model`, `hardware`, `runtime`, `workload`, `quant`, `prompt_tokens`) bằng Linear Regression không?
- Mô hình có khả năng tổng quát hóa sang `model`/`hardware` chưa từng thấy không?
- Giới hạn của việc dự đoán trong phạm vi dữ liệu benchmark là gì?

> Lưu ý: Dự án này là nghiên cứu trên dữ liệu benchmark, không phát triển công cụ xác định khả năng chạy chính xác trên máy tính người dùng tùy ý. Kết luận chỉ áp dụng trong phạm vi phân phối dữ liệu quan sát.

## 2. Nguồn dữ liệu

- **Chính:** `data/raw/llm-speed-runs-2026-10-22.6.csv` (751 rows, 18 columns, CC BY 4.0) từ llm-speed.com
- **Bổ sung (tham khảo):** Một số tệp CSV benchmark bên ngoài (`arynwood_speed.csv`, `llmspeed_runs.csv`, `data_gouv_fr_benchmark_ia_sur_materiel.csv`, `open_llm_benchmarks_and_hardware_ops.csv`) được rà soát trong quá trình làm việc nội bộ — chỉ tham khảo, **không gộp** vào tập huấn luyện baseline; các tệp này không nằm trong gói nộp.

## 3. Phương pháp

- **EDA:** Phân bố, nhóm theo `hardware`/`model`/`runtime`/`workload`, phân tích theo `run_id`. Thêm scatter plot `prompt_tokens` vs `decode_tps` và correlation heatmap chỉ cho numeric có variation (không dùng `batch_size` constant).
- **Feature set:** `model`, `hardware`, `runtime`, `workload`, `quant`, `prompt_tokens`, `batch_size`. `batch_size` là constant (giá trị 1) trong dataset. Không dùng `prefill_tps`, `ttft_ms` làm feature mặc định; không thêm `output_tokens` nếu không có lý do chặt chẽ.
- **Preprocessing:** `ColumnTransformer` với `OneHotEncoder(handle_unknown='ignore')` cho categorical, passthrough numeric, đóng gói trong `Pipeline`.
- **Train/Test split:** `GroupShuffleSplit` theo `run_id` (tránh leakage giữa workload cùng run). `random_state=42`. Không có intersection giữa train/test groups trong split đã kiểm tra.
- **Models:** `DummyRegressor(strategy='median')` (baseline), `LinearRegression`.
- **Metrics:** MAE, RMSE, R².

## 4. Key Findings (dựa trên kết quả đã xác minh)

### Phân bố và mối quan hệ quan sát được
- **Phân bố:** `decode_tps` có phân bố rộng, bị ảnh hưởng mạnh bởi `hardware`, `model`, `runtime`, `workload`.
- **Theo nhóm:** Hiệu năng khác biệt đáng kể giữa các nhóm hardware/model/workload (quan sát từ EDA trong `notebooks/02_data_analysis.ipynb`).
- **Tương quan đơn biến:** `prompt_tokens` có quan hệ với `decode_tps` nhưng không đủ để giải thích toàn bộ biến thiên; correlation chỉ được xem xét cho numeric có variation (theo `reports/drafts/visualization_review.md`).
- **Lưu ý:** Tương quan không đồng nghĩa nhân quả. Nhiều feature có tương quan và high-cardinality.

### So sánh mô hình

| Approach | Model | MAE | RMSE | R² | Status |
|---|---|---|---|---|---|
| A1: quant → `Unknown` | DummyRegressor(median) | 52.687 | 82.290 | -0.205 | VERIFIED (xác minh nội bộ) |
| A1: quant → `Unknown` | LinearRegression | 24.686 | 40.311 | 0.711 | VERIFIED (verify_model_03.py) |
| A2: drop quant NaN | DummyRegressor(median) | (tham khảo) | (tham khảo) | (tham khảo) | PRELIMINARY |
| A2: drop quant NaN | LinearRegression | 27.13 | 44.79 | 0.520 | VERIFIED (verify_model_03.py) |

Ưu tiên **A1 (quant → Unknown)** vì giữ nhiều mẫu hơn và cho kết quả tốt hơn so với A2.

### Generalization Review

Đánh giá với cùng pipeline/A1/seed, dùng `GroupShuffleSplit` với grouping khác nhau:

| Scenario | Model | MAE | RMSE | R² | Notes |
|---|---|---|---|---|---|
| Group by `run_id` (baseline) | DummyRegressor(median) | 52.687 | 82.290 | -0.205 | Train 595/test 156 (268/67 groups), intersection 0 |
| Group by `run_id` (baseline) | LinearRegression | 24.686 | 40.311 | 0.711 | Như trên |
| Unseen `model` | DummyRegressor(median) | 42.378 | 62.866 | -0.045 | Train 604/test 147 (84/22 groups), intersection 0 |
| Unseen `model` | LinearRegression | 59.450 | 68.582 | -0.243 | Kém hơn baseline |
| Unseen `hardware` | DummyRegressor(median) | 67.862 | 84.521 | -0.071 | Train 732/test 19 (14/4 groups), intersection 0 |
| Unseen `hardware` | LinearRegression | 72.133 | 89.259 | -0.194 | **Chỉ 19 mẫu test** (4 hardware chưa thấy) — rất không ổn định |

Nguồn: báo cáo kiểm tra tổng quát hóa nội bộ (đã xác minh; không kèm trong gói nộp).

**Kết luận:** LinearRegression không tổng quát hóa tốt sang `model`/`hardware` chưa từng thấy. Hiệu năng giảm mạnh và thậm chí kém hơn `DummyRegressor`. Phạm vi defendable là nội suy trong phân phối benchmark đã quan sát (cấu hình tương tự), không phải dự đoán cho máy người dùng bất kỳ.

### Permutation Importance
Đã triển khai trong `notebooks/03_model.ipynb` (aggregate theo feature gốc), scoring `neg_mean_absolute_error`, `n_repeats=20`, `random_state=42`. Quy ước dấu đã được xác minh: `perm.importances_mean` là **MAE tăng thêm (TPS)** khi hoán vị feature. Baseline MAE (A1): **24.6859 TPS**. Kết quả đã xác minh:
- model: +53.08 TPS, hardware: +23.15 TPS, quant: +8.28 TPS, workload: +0.69 TPS, runtime: +0.13 TPS, prompt_tokens: +0.021 TPS, batch_size: 0.00 TPS
Error bars là standard deviation qua các lần lặp, không phải confidence interval. Permutation importance và correlation không chứng minh nhân quả. Chi tiết: `reports/drafts/visualization_review.md`, `reports/drafts/model_verification_report.md`. **VERIFIED** (calculation/sign).

## 5. Visualization Strategy

| Biểu đồ | Notebook | Mục đích trả lời câu hỏi nghiên cứu | Ghi chú |
|---|---|---|---|
| Histogram/Boxplot phân bố `decode_tps` theo nhóm | `02_data_analysis.ipynb` | Các yếu tố nào ảnh hưởng đến `decode_tps`? | Quan sát chênh lệch theo `hardware`/`model`/`workload` |
| Scatter `prompt_tokens` vs `decode_tps` | `02_data_analysis.ipynb` | Ảnh hưởng của độ dài prompt? | Chỉ numeric có variation; `batch_size` bị bỏ qua (constant) |
| Correlation heatmap (numeric only) | `02_data_analysis.ipynb` | Mối quan hệ đơn biến giữa numeric features? | Không áp dụng Pearson cho categorical |
| Actual vs Predicted (A1 Dummy/LR) | `03_model.ipynb` | Mô hình có học được pattern không? | So sánh với baseline |
| Residual plot (A1 LR) | `03_model.ipynb` | Phân bố sai số, outlier? | Kiểm tra hệ thống sai lệch |
| MAE/RMSE comparison A1 vs A2 | `03_model.ipynb` | Lựa chọn phương án xử lý `quant` thiếu? | Dẫn đến quyết định ưu tiên A1 |
| Permutation Importance (aggregate by original feature) | `03_model.ipynb` | Feature nào quan trọng tương đối? | Chỉ mang tính tham khảo; cần xác minh, không phải nhân quả |

## 6. Hạn chế

- **Tổng quát hóa:** Không tốt khi dự đoán `model`/`hardware` chưa thấy. Test set `unseen hardware` chỉ có 19 mẫu.
- **Dữ liệu benchmark:** Phân phối không đại diện cho mọi cấu hình máy người dùng.
- **Đặc trưng phần cứng:** `hardware` dạng chuỗi tự do, thiếu thông số cấu trúc (VRAM/RAM/CPU/GPU).
- **Categorical high-cardinality:** Nhiều giá trị `model`/`hardware` hiếm.
- **Feature giới hạn:** `batch_size` constant (1), `model_size` thiếu nhiều (~84.55%), không dùng `prefill_tps/ttft_ms/output_tokens` làm feature mặc định.
- **Phương pháp:** Chỉ Linear Regression + Dummy baseline, chưa tối ưu hyperparameter, chưa thử mô hình phi tuyến.
- **Diễn giải:** Tương quan/importance không phải nhân quả. Grouping theo `run_id` có tính lạc quan so với extrapolation.

## 7. Cấu trúc dự án

```text
SPCK/
├── app.py                          # Streamlit MVP (thành phần bắt buộc)
├── requirements.txt                # Dependencies cho app/inference (đã pin)
├── README.md
├── data/
│   └── raw/
│       └── llm-speed-runs-2026-10-22.6.csv
├── notebooks/
│   ├── 02_data_analysis.ipynb      # EDA
│   └── 03_model.ipynb              # Baseline, LinearRegression, đánh giá
├── docs/
│   ├── DATA_DICTIONARY.md
│   ├── METHODOLOGY.md
│   └── PRESENTATION_NOTES.md       # DRAFT
├── scripts/
│   └── train_deployment_model.py   # Tạo artifact deployment A1
├── src/
│   ├── __init__.py
│   └── inference.py                # Inference module dùng chung (app tái sử dụng)
└── models/
    └── cdk_a1_deployment.joblib    # Artifact deployment (fit full 751 rows)

Ghi chú: Các tệp/thư mục làm việc nội bộ (bộ nhớ dự án, bản nháp, báo cáo xác minh nội bộ,
dữ liệu tham khảo ngoài) không nằm trong gói nộp/deploy này.
```

## 8. Hướng dẫn chạy project

### Yêu cầu
- **Python:** 3.9+ (đã kiểm tra với 3.11.9)
- **Dữ liệu:** Cần có `data/raw/llm-speed-runs-2026-10-22.6.csv`
- **Artifact deployment:** Cần có `models/cdk_a1_deployment.joblib` (phục vụ Streamlit)

### Cài đặt
```bash
pip install -r requirements.txt
```

> Ghi chú: `requirements.txt` phục vụ Streamlit MVP/inference. Để chạy notebook, có thể cần thêm `matplotlib`, `jupyter`/`ipykernel`.

### Chạy Streamlit MVP (thành phần bắt buộc)
Chạy từ **thư mục gốc dự án** (`SPCK/`):
```bash
streamlit run app.py
```

Ứng dụng sử dụng `src/inference.py` và artifact `models/cdk_a1_deployment.joblib` để dự đoán `decode_tps` (token/s). Kết quả chỉ là **ước lượng dựa trên benchmark dataset**; mô hình **chưa tổng quát hóa tốt** trên `model/hardware` chưa từng thấy.

### (Tùy chọn) Tạo lại artifact deployment
```bash
python scripts/train_deployment_model.py
```
Script fit pipeline A1 trên toàn bộ dataset chính theo deployment policy (không thay thế metrics đánh giá holdout A1/A2).

### Chạy notebook phân tích
1. `notebooks/02_data_analysis.ipynb` — EDA và trực quan hóa numeric
2. `notebooks/03_model.ipynb` — Baseline, LinearRegression, đánh giá, permutation importance

### Lưu ý đường dẫn
- Notebook sử dụng đường dẫn tương đối (vd. `../data/raw/...` cho `03_model.ipynb`).
- Một số cấu hình môi trường có thể dùng absolute path đã được thiết lập trước; **không tự ý thay đổi** để tránh làm hỏng khả năng chạy notebook trên máy đã cấu hình.
- Chạy từ thư mục `notebooks/` hoặc đảm bảo working directory đúng để relative path hoạt động.

## 9. Tài liệu bổ sung

- `docs/DATA_DICTIONARY.md`, `docs/METHODOLOGY.md`, `docs/PRESENTATION_NOTES.md` (DRAFT)
- Các báo cáo/bản nháp xác minh nội bộ và tài liệu làm việc dự án không được đưa vào gói nộp này.

## Nguồn xác minh
- Metrics A1/A2 (holdout, theo `run_id`): đã xác minh bằng script kiểm tra nội bộ; số liệu tóm tắt tại bảng ở Mục 4.
- Generalization: đã xác minh bằng đánh giá nội bộ; tóm tắt tại bảng ở Mục 4.
- Split logic: `notebooks/03_model.ipynb` (`GroupShuffleSplit` theo `run_id`, `random_state=42`)
- Data structure: `data/raw/llm-speed-runs-2026-10-22.6.csv`
