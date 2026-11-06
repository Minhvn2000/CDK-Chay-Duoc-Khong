# DATA_DICTIONARY.md

## 1) Nguồn dữ liệu
- **File chính:** `data/raw/llm-speed-runs-2026-10-22.6.csv`
- **Kích thước:** 751 rows × 18 columns
- **License:** CC BY 4.0 (theo ghi nhận trong README)
- **Ghi chú:** Không sửa file raw. Tất cả phân tích đều in-memory.

## 2) Danh sách đầy đủ 18 cột

Dựa trên header thực tế của file CSV:

| Column | Data Type | Meaning | Missing Rate | Role | Used in Model? |
|---|---|---|---|---|---|
| `run_id` | object/string | Định danh benchmark run (nhóm mẫu). Nhiều hàng có thể cùng run. | 0.00% | Grouping key (split) | NO (chỉ dùng cho `GroupShuffleSplit`) |
| `run_url` | object/string | URL liên kết đến run trên llm-speed.com. | 0.00% | Metadata/traceability | NO |
| `date` | object/string | Thời gian ghi nhận benchmark. | 0.00% | Metadata | NO |
| `hardware` | object/string | Mô tả phần cứng (chuỗi tự do). | 0.00% | Categorical feature | YES (input feature) |
| `model` | object/string | Tên model được benchmark. | 0.00% | Categorical feature | YES (input feature) |
| `model_size` | object/string (kèm đơn vị) | Kích thước model (vd. dạng chuỗi). | ~84.55% (635/751) | Exploratory only | NO (quá nhiều missing, không dùng làm feature chính) |
| `quant` | object/string | Kiểu quantization (vd. GGUF quant). | ~36.35% (273/751) | Categorical feature | YES (input feature). Xử lý: A1 → fillna `Unknown`, so sánh với A2 (drop). Ưu tiên A1. |
| `runtime` | object/string | Runtime được dùng (vd. llama.cpp). | 0.00% | Categorical feature | YES (input feature) |
| `runtime_version` | object/string | Phiên bản runtime. | ~26.50% (199/751) | Metadata | NO (không đưa vào feature set hiện tại) |
| `workload` | object/string | Loại workload/đánh giá. | 0.00% | Categorical feature | YES (input feature) |
| `suite_version` | object/string | Phiên bản test suite. | 0.00% | Metadata | NO |
| `decode_tps` | float/numeric | Tốc độ sinh token (tokens/s). | 0.00% | **Target variable** | TARGET (không phải feature) |
| `prefill_tps` | float/numeric | Tốc độ prefill. | ~41.28% (310/751) | Speed metric (related) | NO (không dùng làm feature mặc định theo quy định) |
| `ttft_ms` | float/numeric | Time to first token (ms). | ~18.24% (137/751) | Latency metric | NO (không dùng làm feature mặc định theo quy định) |
| `prompt_tokens` | int/numeric | Số token của prompt. | 0.00% | Numeric feature | YES (input feature) |
| `output_tokens` | int/numeric | Số token sinh ra. | 0.00% | Output-related | NO (không dùng làm feature mặc định theo quy định) |
| `batch_size` | int/numeric | Batch size. | 0.00% | Numeric feature | YES (input feature). **Constant = 1** trong dataset → không đủ variation cho scatter/correlation analysis theo yêu cầu. |
| `source` | object/string | Nguồn dữ liệu (community/llm-speed). | 0.00% | Metadata (bias indicator) | NO |

## 3) Phân loại theo vai trò

### Raw dataset columns
Tất cả 18 cột trên đều là cột trong file raw.

### Selected input features (pipeline input)
**7 input columns** dùng trực tiếp cho `X`:
- `model` (categorical)
- `hardware` (categorical)
- `runtime` (categorical)
- `workload` (categorical)
- `quant` (categorical) — xử lý missing bằng `Unknown` trong A1
- `prompt_tokens` (numeric)
- `batch_size` (numeric) — constant

> **Quan trọng:** Số lượng feature sau `OneHotEncoder` sẽ lớn hơn 7 (tùy thuộc vào cardinality của các categorical). Không nhầm 7 input columns với số feature sau encoding.

### Target variable
- `decode_tps` — biến cần dự đoán (tokens/s).

### Columns excluded from modeling
- `run_id` — chỉ dùng cho grouping/split
- `run_url`, `date`, `runtime_version`, `suite_version`, `source` — metadata
- `model_size` — missing quá nhiều (~84.55%)
- `prefill_tps`, `ttft_ms`, `output_tokens` — không đưa vào feature set mặc định (tránh leakage/diễn giải nhầm)

### Transformed features sau OneHotEncoder
Sau `ColumnTransformer`:
- Categorical (`model`, `hardware`, `runtime`, `workload`, `quant`) được one-hot encode bởi `OneHotEncoder(handle_unknown='ignore')` → tạo nhiều dummy columns.
- Numeric (`prompt_tokens`, `batch_size`) được passthrough (không scale trong pipeline hiện tại).
- Tổng số feature = (số cột one-hot) + 2. Giá trị cụ thể phụ thuộc vào vocabulary sau fit, không cố định bằng 7.

## 4) Lý do các quyết định quan trọng

- **`decode_tps` là target:** Phù hợp câu hỏi "chạy được không" — đo tốc độ sinh token thực tế.
- **Chọn 7 input columns:** Dựa trên EDA và chiến lược dữ liệu (`agent_workspace/reports/dataset_strategy.md`), đảm bảo đơn giản, tránh leakage.
- **`batch_size` constant:** Trong dataset hiện tại `batch_size == 1` cho tất cả mẫu → không có variance, không dùng cho scatter/correlation theo yêu cầu.
- **`model_size` missing ~84.55%:** Quá nhiều missing, khó xử lý mà không thêm giả định mạnh; không đưa vào feature set chính.
- **`prefill_tps`, `ttft_ms`, `output_tokens` không dùng làm feature:** Tránh leakage hoặc nhầm lẫn ngữ cảnh dự đoán. Chỉ dùng để quan sát nếu cần, không làm input mặc định.
- **Xử lý `quant`:** So sánh A1 (`fillna('Unknown')`) vs A2 (dropna). A1 cho kết quả tốt hơn (R² 0.711 vs 0.520) và giữ nhiều mẫu hơn → được ưu tiên.

## 5) Nguồn xác minh
- Header & missing: đọc trực tiếp `data/raw/llm-speed-runs-2026-10-22.6.csv`
- Metrics A1/A2: `agent_workspace/scripts/verify_model_03.py`
- Feature set: `notebooks/03_model.ipynb`
- Quyết định: `AGENTS.md`, `DECISIONS.md`
