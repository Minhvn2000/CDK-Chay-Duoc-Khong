# Ghi chú trình bày (PRESENTATION_NOTES) – CDK – Chạy Được Không

**Trạng thái:** DRAFT
**Mục đích:** Tài liệu hỗ trợ trình bày báo cáo cuối khóa. Chỉ sử dụng thông tin đã được xác minh.

## 1) Tổng quan bài toán và mục tiêu nghiên cứu

### Bài toán
Dự án nghiên cứu khả năng dự đoán tốc độ sinh token (`decode_tps`, token/s) của các mô hình ngôn ngữ lớn (LLM) chạy cục bộ (local) dựa trên cấu hình benchmark.

### Mục tiêu
- Xác định các yếu tố ảnh hưởng đến `decode_tps` trong tập dữ liệu benchmark.
- Đánh giá khả năng dự đoán `decode_tps` bằng mô hình **Hồi quy tuyến tính (Linear Regression)**.
- Kiểm tra khả năng tổng quát hóa sang `model` và `hardware` chưa từng thấy.
- Chỉ ra rõ giới hạn của mô hình trong thực tế.

### Lưu ý quan trọng về phạm vi
- Đây là nghiên cứu trên **dữ liệu benchmark**, không phải công cụ xác định chắc chắn một máy tính bất kỳ có "chạy được" LLM hay không.
- Kết luận chỉ có giá trị trong phạm vi phân phối dữ liệu đã quan sát.

## 2) Mô tả dataset và chất lượng dữ liệu

### Thông tin chung
- **Tập dữ liệu chính:** `data/raw/llm-speed-runs-2026-10-22.6.csv`
- **Kích thước:** 751 hàng × 18 cột
- **Số `run_id` khác nhau:** 335
- **Biến mục tiêu:** `decode_tps` (token/s)
- **Giấy phép:** CC BY 4.0

### Đánh giá chất lượng
- `decode_tps`: đầy đủ (không missing)
- `model_size`: thiếu khoảng **84,55%** → không đưa vào mô hình chính
- `quant`: thiếu khoảng **36,35%** → xử lý bằng cách so sánh A1/A2
- `batch_size`: có giá trị **không đổi = 1** trong toàn bộ tập dữ liệu
- `hardware`: lưu dưới dạng chuỗi mô tả tự do, thiếu thông số cấu trúc (VRAM/RAM/CPU/GPU)
- Nhiều đặc trưng phân loại có **số lượng giá trị cao (high-cardinality)**

### Tương quan đơn biến
- **Pearson correlation** giữa `prompt_tokens` và `decode_tps`: **r ≈ 0.162**, **n = 751**
- Mối tương quan này **yếu và dương**. **Tương quan không đồng nghĩa với quan hệ nhân quả.**

## 3) Các quyết định tiền xử lý và lựa chọn feature

### Feature đầu vào (7 cột)
Mô hình sử dụng 7 đặc trưng gốc:
- Phân loại: `model`, `hardware`, `runtime`, `workload`, `quant`
- Số: `prompt_tokens`, `batch_size`

> **Lưu ý:** Sau khi áp dụng `OneHotEncoder` cho các đặc trưng phân loại, số lượng feature thực tế sẽ **lớn hơn 7**. Không nhầm 7 cột đầu vào với số feature sau mã hóa.

### Các quyết định quan trọng
- **Không dùng làm feature mặc định:** `prefill_tps`, `ttft_ms`, `output_tokens` (tránh leakage/diễn giải nhầm)
- **Không dùng `model_size`:** do tỷ lệ thiếu quá cao (~84,55%)
- **Xử lý `quant` thiếu:** So sánh A1 (thay bằng `Unknown`) và A2 (loại bỏ hàng). Ưu tiên **A1** vì giữ được nhiều mẫu hơn và cho kết quả tốt hơn
- **`batch_size`:** constant = 1 → không có biến thiên, không dùng cho phân tích tương quan

### Tiền xử lý
- `ColumnTransformer` với `OneHotEncoder(handle_unknown='ignore')` cho đặc trưng phân loại
- `passthrough` cho đặc trưng số
- Đóng gói trong `Pipeline` để đảm bảo xử lý nhất quán giữa train/test

## 4) Những phát hiện EDA quan trọng

- `decode_tps` có phân bố rộng và bị ảnh hưởng đáng kể bởi `hardware`, `model`, `runtime`, `workload`
- Có sự khác biệt rõ rệt về tốc độ giữa các nhóm phần cứng, mô hình và workload
- `prompt_tokens` có tương quan yếu với `decode_tps` (r ≈ 0.162) → không đủ để giải thích toàn bộ biến thiên
- Nhiều đặc trưng phân loại có số lượng giá trị cao và có khả năng tương quan với nhau
- Phân tích theo `run_id` cho thấy nhiều benchmark đến từ cùng một run nhưng khác nhau về `workload`

## 5) Thiết kế Linear Regression và cách chia train/test

### Mô hình
- **Baseline:** `DummyRegressor(strategy='median')`
- **Mô hình chính:** `LinearRegression` (không tối ưu hyperparameter)

### Chia dữ liệu
- Sử dụng **`GroupShuffleSplit` theo `run_id`** với `test_size=0.2`, `random_state=42`
- Đảm bảo **không có giao thoa (intersection = 0)** giữa nhóm `run_id` trong train và test → hạn chế leakage giữa các workload thuộc cùng một run
- Cách chia này vẫn chỉ đánh giá trong phạm vi phân phối benchmark, **không đồng nghĩa** với khả năng tổng quát hóa sang model/hardware hoàn toàn mới

## 6) Kết quả đánh giá A1/A2 so với Dummy baseline

Các kết quả dưới đây đã được xác minh bằng script `agent_workspace/scripts/verify_model_03.py`:

| Phương án | Mô hình | MAE | RMSE | R² |
|---|---|---|---|---|
| **A1 (quant → Unknown)** | DummyRegressor (median) | 52.6869 | 82.2901 | -0.2053 |
| **A1 (quant → Unknown)** | LinearRegression | **24.6859** | **40.3109** | **0.7108** |
| **A2 (loại bỏ hàng có quant thiếu)** | DummyRegressor (median) | 58.6658 | 81.2513 | -0.5809 |
| **A2 (loại bỏ hàng có quant thiếu)** | LinearRegression | 27.1261 | 44.7867 | 0.5197 |

**Kết luận so sánh:** LinearRegression vượt trội hơn baseline đáng kể ở cả hai phương án. **Ưu tiên A1** vì giữ được nhiều mẫu hơn và cho kết quả tốt hơn (R² 0.711 so với 0.520).

## 7) Permutation Importance và cách diễn giải

### Cấu hình
- Sử dụng pipeline A1 đã fit trên tập train
- `X_test` gồm 7 đặc trưng gốc
- `scoring="neg_mean_absolute_error"`, `n_repeats=20`, `random_state=42`
- Tính theo **đặc trưng gốc** (gộp thay vì tách từng dummy column)
- Đơn vị: **Mức tăng MAE (token/s)**

### Kết quả đã xác minh
| Đặc trưng | Mức tăng MAE trung bình (token/s) | Độ lệch chuẩn |
|---|---|---|
| `model` | +53.08 | 3.51 |
| `hardware` | +23.15 | 2.46 |
| `quant` | +8.28 | 0.98 |
| `workload` | +0.69 | 0.20 |
| `runtime` | +0.13 | 0.295 |
| `prompt_tokens` | +0.021 | 0.018 |
| `batch_size` | 0.00 | 0.000 |

### Cách diễn giải đúng
- **Giá trị dương (+):** Khi hoán vị (xáo trộn) đặc trưng đó, MAE trên tập test **tăng lên** → mô hình phụ thuộc vào đặc trưng này cho việc dự đoán.
- **Gần 0:** Ảnh hưởng rất nhỏ hoặc không đáng kể.
- **Giá trị âm (-) (nếu xuất hiện):** Hoán vị có thể làm MAE **giảm** trên tập test này. Điều này có thể do nhiễu mẫu, đặc trưng tương quan, dư thừa thông tin hoặc tổ hợp không thực tế. **Không** tự động hiểu là đặc trưng có hại.
- **Error bar:** Là **độ lệch chuẩn** qua các lần hoán vị, **không phải** khoảng tin cậy (confidence interval).
- **Quan trọng:** *Permutation Importance* đo **mức quan trọng dự đoán (predictive importance)**, không phải *tác động nhân quả (causal effect)*. **Độ quan trọng dự đoán không chứng minh quan hệ nhân quả.**

## 8) Hạn chế của mô hình

### Hạn chế về tổng quát hóa
- **Không tổng quát hóa tốt** sang `model` hoặc `hardware` chưa từng thấy (kết quả từ generalization review cho thấy hiệu năng kém hơn cả baseline)
- **Tập test "unseen hardware" chỉ có 19 mẫu** (4 hardware) → các chỉ số không ổn định
- Phạm vi có thể bảo vệ được chỉ là **nội suy** trong phân phối benchmark đã quan sát, không phải dự đoán cho mọi máy tính người dùng

### Hạn chế về dữ liệu và phương pháp
- `hardware` dạng chuỗi tự do, thiếu thông số cấu trúc
- Nhiều đặc trưng phân loại có high-cardinality
- `batch_size` không đổi (1), `model_size` thiếu nhiều
- Chỉ sử dụng `Linear Regression` + `DummyRegressor` baseline, chưa thử mô hình phi tuyến
- Nhiều đặc trưng có khả năng tương quan với nhau → ảnh hưởng đến diễn giải permutation importance
- Chỉ đánh giá trên dữ liệu benchmark, không đại diện cho toàn bộ môi trường sử dụng thực tế
- **Tương quan ≠ nhân quả**, **độ quan trọng dự đoán ≠ tác động nhân quả**

## 9) Kết luận: mô hình làm được gì và không làm được gì

### Mô hình CÓ THỂ làm được
- Dự đoán `decode_tps` tương đối tốt trong phạm vi cấu hình benchmark **tương tự** với những gì đã quan sát trong tập dữ liệu
- Chỉ ra rõ các yếu tố có liên quan đến tốc độ sinh token trong điều kiện benchmark
- Vượt trội hơn baseline một cách rõ rệt khi chia theo `run_id`

### Mô hình KHÔNG THỂ làm được
- **Không** đảm bảo dự đoán chính xác cho `model` hoặc `hardware` hoàn toàn mới
- **Không** phải là công cụ để khẳng định chắc chắn một PC bất kỳ có "chạy được" LLM hay không
- **Không** chứng minh quan hệ nhân quả giữa các đặc trưng và `decode_tps`
- **Không** khái quát hóa tốt ra ngoài phân phối benchmark hiện tại
- **Không** thay thế việc đánh giá thực tế trên máy tính mục tiêu

**Kết luận ngắn gọn:** Linear Regression với feature set hiện tại hoạt động hữu ích cho mục đích **phân tích và so sánh trong phạm vi benchmark**, nhưng cần cực kỳ thận trọng khi áp dụng ra ngoài phạm vi đó.

## Ghi chú kiểm chứng

- **Trạng thái runtime notebook:** Hai notebook `notebooks/02_data_analysis.ipynb` và `notebooks/03_model.ipynb` **chưa được chạy thành công từ đầu đến cuối** trên kernel mới. Không tuyên bố trạng thái PASS.
- **Tất cả số liệu trên:** Đều dựa trên kết quả đã xác minh (`agent_workspace/scripts/verify_model_03.py`, `agent_workspace/reports/model_generalization_review.md`, báo cáo kiểm chứng permutation importance).
- **Tài liệu này:** DRAFT – chỉ sử dụng thông tin có bằng chứng, không thêm insight chưa xác minh.
