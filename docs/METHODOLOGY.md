# METHODOLOGY.md

Tài liệu này mô tả quy trình thực hiện dự án CDK – Chạy Được Không dựa trên thực tế notebook, script verification và báo cáo nội bộ.

## 1) Research questions
- Các yếu tố nào ảnh hưởng đến `decode_tps` trong tập dữ liệu benchmark?
- Có thể dự đoán `decode_tps` từ cấu hình benchmark bằng Linear Regression không?
- Mô hình có khả năng tổng quát hóa sang `model`/`hardware` chưa từng thấy không?
- Giới hạn ứng dụng thực tế là gì?

**Tại sao:** Xác định rõ phạm vi nghiên cứu trên dữ liệu benchmark, tránh tuyên bố công cụ dự đoán tổng quát.

## 2) Data collection
- **Nguồn chính:** llm-speed.com, file `data/raw/llm-speed-runs-2026-10-22.6.csv` (CC BY 4.0).
- **Nguồn tham khảo:** `data/external/*`, `agent_workspace/external_kaggle/*` chỉ để quan sát, không gộp vào training.

**Tại sao:** Sử dụng dataset có cấu trúc benchmark rõ ràng (run_id, hardware, model, runtime, workload, metrics).
**Giải quyết:** Có baseline nhất quán.
**Giới hạn:** Chỉ đại diện cho cấu hình benchmark, không phủ hết máy người dùng.

## 3) Dataset selection
Chọn dataset chính 751 rows × 18 columns làm tập duy nhất cho baseline modeling. Các dataset khác không được append.

**Tại sao:** Tránh schema mismatch, rủi ro bias/leakage, khó đảm bảo đơn vị đo nhất quán.
**Giải quyết:** Giữ pipeline đơn giản, dễ bảo vệ.
**Giới hạn:** Giảm kích thước training, giới hạn tổng quát hóa.

## 4) Data quality audit
Thực hiện trong `notebooks/02_data_analysis.ipynb`:
- Kiểm tra shape, dtypes, missing values
- Phân tích `run_id` (group size, variation)
- Kiểm tra duplicates
- Kiểm tra `batch_size` (unique/constant)
- Kiểm tra `model_size`, `quant`, metrics
- Phân bố theo `source`/`runtime`/`workload`/`hardware`

**Tại sao:** Hiểu cấu trúc trước khi feature selection.
**Giải quyết:** Phát hiện `batch_size` constant, `model_size` thiếu nhiều, `quant` thiếu.
**Giới hạn:** Không loại bỏ outlier một cách tùy tiện (theo ghi nhận trong notebook).

## 5) Missing value handling
- **`quant`:** So sánh A1 (fillna `Unknown`) vs A2 (dropna). Ưu tiên A1 (R² 0.711 vs 0.520).
- **`model_size`:** Không impute, không dùng làm feature (missing ~84.55%).
- **`runtime_version`, `prefill_tps`, `ttft_ms`:** Chỉ quan sát, không dùng làm feature mặc định.
- **Target `decode_tps`:** 0 missing, dropna không cần thiết.

**Tại sao:** Cân bằng giữa giữ mẫu và tránh thêm giả định không có cơ sở.
**Giải quyết:** Quyết định có thể bảo vệ bằng so sánh metrics.
**Giới hạn:** Impute `Unknown` không phản ánh đúng quant thực tế.

## 6) EDA
Trong `notebooks/02_data_analysis.ipynb`:
- Thống kê mô tả `decode_tps`
- Phân bố theo runtime/workload/source
- Top hardware
- Phân tích repeated `run_id`
- Scatter `prompt_tokens` vs `decode_tps`
- Correlation heatmap (numeric only)

**Tại sao:** Trả lời câu hỏi nghiên cứu về yếu tố ảnh hưởng.
**Giải quyết:** Xác định feature set hợp lý.
**Giới hạn:** Chỉ quan sát, không suy ra nhân quả.

## 7) Visualization selection và lý do
Xem bảng trong `README.md` (Visualization Strategy). Nguyên tắc:
- Chọn biểu đồ để trả lời câu hỏi nghiên cứu cụ thể
- Chỉ dùng Pearson cho numeric có variation
- Không vẽ correlation cho categorical
- Giữ đơn giản, dễ diễn giải

**Tại sao:** Tránh biểu đồ gây hiểu nhầm.
**Giới hạn:** Một số quan hệ phức tạp không hiện rõ bằng biểu đồ đơn biến.

## 8) Feature selection
**Input columns (7):** `model`, `hardware`, `runtime`, `workload`, `quant`, `prompt_tokens`, `batch_size`
**Loại bỏ:** `prefill_tps`, `ttft_ms`, `output_tokens`, `model_size`, metadata

**Tại sao:** Dựa trên availability, tránh leakage, phù hợp với câu hỏi dự đoán từ cấu hình benchmark.
**Giải quyết:** Feature set nhất quán với `AGENTS.md`.
**Giới hạn:** Bỏ qua thông tin quan trọng có thể có (vd. thông số phần cứng cấu trúc).

## 9) OneHotEncoder và ColumnTransformer
- `OneHotEncoder(handle_unknown='ignore')` cho 5 categorical
- `passthrough` cho 2 numeric
- `remainder='drop'`
- Đóng gói trong `Pipeline`

**Tại sao:** Xử lý categorical không thứ tự, an toàn với unseen categories trong predict (dù gây vấn đề khi generalize).
**Giải quyết:** Pipeline tái sử dụng nhất quán giữa notebook và script.
**Giới hạn:** Tạo không gian feature cao, khó diễn giải trực tiếp, unseen category → zero vector.

## 10) Pipeline
Sử dụng `sklearn.pipeline.Pipeline` kết hợp `ColumnTransformer` + estimator.

**Tại sao:** Đảm bảo preprocessing đồng nhất train/test, tránh data leakage.
**Giải quyết:** Code rõ ràng, dễ reproduce.
**Giới hạn:** Một số phân tích (permutation importance) cần cẩn thận khi aggregate.

## 11) GroupShuffleSplit theo run_id
`GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)` với `groups = run_id`. Kiểm tra intersection train/test groups = 0.

**Tại sao:** Tránh leakage giữa workload cùng benchmark run.
**Giải quyết:** Đánh giá chặt chẽ hơn so với random split.
**Giới hạn:** Vẫn là hold-out trong phân phối benchmark; không tương đương với unseen model/hardware hoàn toàn mới.

## 12) DummyRegressor và LinearRegression
- **DummyRegressor(strategy='median'):** Baseline đơn giản
- **LinearRegression:** Mô hình tuyến tính, dễ diễn giải, phù hợp với giới hạn task

**Tại sao:** So sánh với baseline để chứng minh có thông tin hữu ích.
**Giải quyết:** Kết quả dễ bảo vệ.
**Giới hạn:** Linear assumption, không bắt được tương tác phức tạp tốt.

## 13) MAE, RMSE và R²
- **MAE:** Trung bình sai số tuyệt đối (dễ hiểu, đơn vị tokens/s)
- **RMSE:** Phạt outlier mạnh hơn
- **R²:** Tỷ lệ phương sai giải thích được (so với mean)

**Tại sao:** Cùng nhau đánh giá cả độ chính xác và sai số lớn.
**Giải quyết:** Đầy đủ cho mục đích báo cáo.
**Giới hạn:** R² có thể gây hiểu nhầm, đặc biệt với generalization kém.

## 14) Generalization review theo run_id, model và hardware
Thực hiện bởi `agent_workspace/scripts/generalization_review.py`, kết quả trong `agent_workspace/reports/model_generalization_review.md`:
- Group by `run_id` (baseline)
- Unseen `model`
- Unseen `hardware` (test set chỉ 19 rows, 4 hardware)

**Tại sao:** Kiểm tra giả định "có thể tổng quát hóa" một cách thực tế.
**Giải quyết:** Chỉ ra rõ giới hạn phạm vi sử dụng.
**Giới hạn:** Test set unseen hardware rất nhỏ → metrics không ổn định.

## 15) Limitations
- **Tổng quát hóa:** Không tốt cho model/hardware chưa thấy. Chỉ defendable trong phạm vi tương tự benchmark.
- **Dữ liệu:** Hardware dạng chuỗi, thiếu thông số cấu trúc; high-cardinality; `batch_size` constant.
- **Phương pháp:** Chỉ Linear Regression, chưa tuning; correlation/importance không phải nhân quả.
- **Permutation importance:** Đã triển khai nhưng dấu âm cần xác minh thủ công → chưa đưa vào kết luận chính thức.
- **Generalization nhỏ:** Unseen hardware chỉ 19 mẫu.

## Nguồn xác minh
- `notebooks/02_data_analysis.ipynb`, `notebooks/03_model.ipynb`
- `agent_workspace/scripts/verify_model_03.py`
- `agent_workspace/scripts/generalization_review.py`
- `agent_workspace/reports/model_generalization_review.md`
- `data/raw/llm-speed-runs-2026-10-22.6.csv`
- `README.md`, `AGENTS.md`, `DECISIONS.md`, `PROJECT_STATE.md`
