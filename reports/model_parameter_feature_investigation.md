# CDK — Nghiên cứu Feature Engineering: Trích xuất số lượng tham số từ tên Model

**Ngày:** 2026-11-07  
**Trạng thái:** Research-only. Không thay đổi production model, artifact, dataset, `app.py`, `src/inference.py`, notebook hay kết quả A1 đã công bố.  
**Dataset:** `data/raw/llm-speed-runs-2026-10-22.6.csv` — 751 rows, 18 cột, 106 model names, 335 `run_id`.  
**SHA-256 dataset:** `0b7a57d6f2a8ef768f161df143d665c2a3723dfb21b426a289ea4e062f522eb2`  
**Artifact tham chiếu:** `models/cdk_a1_deployment.joblib` — SHA-256 `b112a9bada0383e134c32a055f82e0c2b40e1a26c50d0a5bd55aa8bbd70d192d` (không load/ghi đè trong nghiên cứu này).  
**Môi trường:** Python 3.11.9, pandas 3.0.3, numpy 2.4.6, scikit-learn 1.9.0, joblib 1.5.3.

> Báo cáo này chỉ trả lời câu hỏi nghiên cứu. Không có khuyến nghị nào được tự động triển khai vào CDK.

---

## 0. Tóm tắt kết luận

| Câu hỏi | Kết luận ngắn | Bằng chứng chính |
|---|---|---|
| Trích xuất `model_params_b` có đáng tin cậy? | **Có, nhưng không đầy đủ.** Tin cậy với tên có token kích thước rõ ràng; 23/106 tên không có thông tin. | 83/106 model (78,30%); 606/751 rows (80,69%); 25/25 synthetic test pass; 5/5 model có `model_size` khớp chính xác. |
| Biến thể B (A1 + `model_params_b`) có cải thiện holdout chuẩn? | **Không có cải thiện có ý nghĩa.** | ΔMAE = −0,016 ± 0,307 (20 seeds); 9/20 seed tốt hơn; MAE seed 42: 24,7292 so với 24,6859. |
| Biến thể C (thay `model` bằng `model_params_b`) có cải thiện holdout chuẩn? | **Không, tệ hơn rõ rệt.** | MAE 59,5196 so với 24,6859; 0/20 seed tốt hơn; ΔMAE = +32,600 ± 4,650. |
| B hoặc C có cải thiện tổng quát hóa sang model chưa thấy? | **Không ổn định.** B tệ hơn trung bình; C tệ hơn trung bình dù có 7/20 seed tốt hơn. | B: +3,236 MAE; C: +5,167 MAE; seed 42 C tốt hơn nhưng là ngoại lệ. |
| Có giảm dự đoán âm không? | **Không nhất quán.** C giảm âm trên `run_id` nhưng tăng mạnh trên unseen-model. | `run_id`: A 10, B 10, C 2; unseen-model: A 8, B 17, C 43 (tổng 20 seeds). |
| `model_params_b` bổ sung hay thay thế được `model`? | **Bổ sung rất ít; không thể thay thế.** | Permutation importance: `model` +50,80 MAE so với `model_params_b` +0,49 MAE (run_id, seed 42). |
| Có đủ bằng chứng để đổi feature engineering của CDK? | **Chưa.** | Không biến thể nào vượt A1 một cách ổn định trên cả hai protocol. |

**Khuyến nghị tổng:** Giữ A1 làm production model. Không triển khai B hoặc C. Hướng `active_params_b` chỉ nên được coi là giả thuyết nghiên cứu tiếp theo, không phải thay đổi production.

---

## 1. Tóm tắt nghiên cứu

### 1.1. Bối cảnh

- `model_size` thiếu 635/751 rows (84,55%).
- A1 dùng `model` như categorical feature; OneHotEncoder coi mỗi tên model là một category riêng, không hiểu quan hệ thứ tự/kích thước giữa 7B và 70B.
- Giả thuyết: bổ sung `model_params_b` (số tỷ tham số, đơn vị B) dưới dạng numeric feature có thể giúp Linear Regression học quan hệ giữa quy mô model và `decode_tps`, cải thiện tổng quát hóa sang model chưa từng thấy.

### 1.2. Phạm vi

- Chỉ nghiên cứu trên bản sao dữ liệu trong bộ nhớ/temp; không ghi vào `data/raw/`.
- Không retrain, không ghi đè artifact production.
- Không sửa `app.py`, `src/inference.py`, notebook, `requirements.txt`.
- Không commit/push.

### 1.3. Kết quả cốt lõi

1. Extractor hoạt động tốt với các tên có token rõ ràng, kể cả MoE dạng `40B-A2.5B`, nhưng không thể suy ra tham số cho 23 tên không có token.
2. Thêm `model_params_b` vào A1 (biến thể B) gần như không thay đổi holdout chuẩn và không cải thiện unseen-model một cách ổn định.
3. Thay `model` bằng `model_params_b` (biến thể C) làm mất thông tin định danh model; holdout chuẩn tệ hơn khoảng 2,4 lần MAE.
4. `model_params_b` có tương quan nghịch với `decode_tps`, nhưng tương quan yếu hơn nhiều so với khả năng phân biệt của `model`; đây là quan hệ thống kê, không phải nhân quả.
5. Không có bằng chứng đủ mạnh để thay đổi feature engineering của CDK.

---

## 2. Phương pháp trích xuất tham số

### 2.1. Quy tắc

Extractor chỉ đọc chuỗi `model`, không dùng `model_size` để điền giá trị. Ba regex chính:

```python
# Tổng số tham số: số + B/b, không dính vào chữ/số/dấu chấm phía trước.
TOTAL_RE = re.compile(r"(?<![A-Za-z0-9.])(\d+(?:\.\d+)?)\s*[Bb]\b")

# Active parameters của MoE: A2.5B, A42B, A9B, A3B.
ACTIVE_RE = re.compile(r"(?<![A-Za-z0-9])[Aa](\d+(?:\.\d+)?)\s*[Bb]\b")

# MoE dạng tích: 8x7B. Không quy đổi thành 56B.
MOE_X_RE = re.compile(r"(?<![A-Za-z0-9])(\d+(?:\.\d+)?)\s*[xX]\s*(\d+(?:\.\d+)?)\s*[Bb]\b")
```

Quy tắc quyết định:

| Trường hợp | Cách xử lý |
|---|---|
| Một token tổng duy nhất, không có active token | `status="dense"`, `model_params_b = total` |
| Một token tổng + một active token | `status="moe_explicit_total"`, lưu cả total và active |
| Chỉ có `AxB` không có tổng | `status="ambiguous"`, **không** quy đổi |
| Nhiều token tổng khác nhau | `status="ambiguous"` |
| Không có token nào | `status="unknown"`, `model_params_b = None` |
| `ctx16k`, `context-32k`, `v0.1`, `Q4_K_M`, `IQ3_XXS`, `00001-of-00005`, `24GB` | Không khớp regex → unknown |
| Chữ hoa/thường, số thập phân | Hỗ trợ (`14B`, `14b`, `0.5B`, `1.5B`) |

### 2.2. Kiểm thử hợp đồng (synthetic)

25/25 test pass, gồm:

- Dense: `0.5B`, `1.5B`, `7B`, `14B`, `30B`, `70B`, `14b`.
- Không nhầm: `ctx16k`, `context-32k`, `v0.1`, `Q4_K_M`, `IQ3_XXS`, `00001-of-00005`, `24GB`.
- MoE có tổng rõ: `Flint-40B-A2.5B` → 40/2,5; `LFM2.5-24B-A2B` → 24/2; `Muse-Spark-Open-816B-A42B` → 816/42; `NVIDIA-Nemotron-3.5-Bolt-74B-A9B` → 74/9.
- MoE không rõ: `8x7B`, `MoE-8x7B`, `A2.5B` → `ambiguous`, không tự quy đổi.

### 2.3. So sánh với `model_size`

- `model_size` có 116/751 rows (15,45%), 17 tên model khác nhau.
- Trong 17 tên đó, chỉ 5 tên vừa có `model_size` vừa trích xuất được từ tên; cả 5 khớp chính xác trong 0,1B:
  - `lmstudio-community-Qwen3-Next-80B-A3B-Instruct-MLX-4bit`: 80B ↔ 80,0
  - `mlx-community/Qwen2.5-0.5B-Instruct-4bit`: 0.5B ↔ 0,5
  - `mlx-community/Qwen2.5-14B-Instruct-4bit`: 14B ↔ 14,0
  - `mlx-community/Qwen2.5-32B-Instruct-4bit`: 32B ↔ 32,0
  - `mlx-community/Qwen2.5-7B-Instruct-4bit`: 7B ↔ 7,0
- 104 rows có `model_size` nhưng tên không trích xuất được (ví dụ `gemma3`, `deepseek-r1`, `gpt-oss`).
- 3 tên model có nhiều hơn một giá trị `model_size` trong dataset: `gemma3` (4 giá trị), `deepseek-r1` (3), `qwen2.5-coder` (2). Vì vậy `model_size` không phải lúc nào cũng là thuộc tính cố định theo tên model; không dùng nó làm fallback trong nghiên cứu này.

---

## 3. Kết quả kiểm tra chất lượng trích xuất

### 3.1. Theo model name

| Nhóm | Số model | Tỷ lệ |
|---|---:|---:|
| Tổng số model name | 106 | 100,00% |
| Trích xuất thành công | 83 | 78,30% |
| — Dense | 66 | 62,26% |
| — MoE có tổng rõ | 17 | 16,04% |
| Không trích xuất được (`unknown`) | 23 | 21,70% |
| `ambiguous` trong dataset | 0 | 0,00% |

### 3.2. Theo observation

| Nhóm | Số rows | Tỷ lệ |
|---|---:|---:|
| Tổng rows | 751 | 100,00% |
| Có `model_params_b` | 606 | 80,69% |
| Thiếu `model_params_b` | 145 | 19,31% |
| — Dense rows | 470 | 62,58% |
| — MoE rows | 136 | 18,11% |
| — Unknown rows | 145 | 19,31% |

### 3.3. Phân bố giá trị trích xuất

24 giá trị khác nhau: 0,5; 0,6; 3; 4; 7; 8; 9; 12; 14; 15; 20; 22; 24; 27; 30; 31; 32; 35; 40; 70; 72; 74; 80; 816.

Các giá trị phổ biến nhất: 30B (16 model), 31B (8), 27B (7), 32B (7), 40B (5), 8B (5).

### 3.4. Ví dụ đại diện

| Nhóm | Model name | `model_params_b` | `active_params_b` |
|---|---|---:|---:|
| Dense | `Llama-3.3-70B-Instruct` | 70,0 | 70,0 |
| Dense | `mlx-community/Qwen2.5-0.5B-Instruct-4bit` | 0,5 | 0,5 |
| Dense | `Qwen3.6-27B-Q4_K_M.gguf` | 27,0 | 27,0 |
| MoE | `Flint-40B-A2.5B-UD-ctx8k` | 40,0 | 2,5 |
| MoE | `LFM2.5-24B-A2B` | 24,0 | 2,0 |
| MoE | `Muse-Spark-Open-816B-A42B-UD-00001-of-00005` | 816,0 | 42,0 |
| MoE | `Qwen3-Coder-30B-A3B-Instruct` | 30,0 | 3,0 |
| Unknown | `gemma3` | — | — |
| Unknown | `deepseek-r1` | — | — |
| Unknown | `Sage-A1.5-ctx8k` | — | — |
| Unknown | `phi-4` | — | — |

---

## 4. Các trường hợp không rõ ràng và MoE

### 4.1. `ambiguous`

- Trong 106 tên model của dataset: **0** trường hợp `ambiguous`.
- Trong synthetic test: `8x7B`, `MoE-8x7B`, `A2.5B` được đánh dấu `ambiguous`; extractor **không** tự quy đổi `8x7B` thành 56B.
- Lý do: `8x7B` có thể là 8 expert × 7B, nhưng tổng tham số thực tế còn phụ thuộc shared layers/router; quy đổi đơn giản sẽ sai.

### 4.2. 17 model MoE được nhận diện

| Model | Total (B) | Active (B) |
|---|---:|---:|
| `Flint-40B-A2.5B-NVFP4` | 40 | 2,5 |
| `Flint-40B-A2.5B-UD-cpumoe12-ctx8k` | 40 | 2,5 |
| `Flint-40B-A2.5B-UD-cpumoe6-ctx8k` | 40 | 2,5 |
| `Flint-40B-A2.5B-UD-ctx16k` | 40 | 2,5 |
| `Flint-40B-A2.5B-UD-ctx8k` | 40 | 2,5 |
| `LFM2.5-24B-A2B` | 24 | 2 |
| `Muse-Spark-Open-816B-A42B-UD-00001-of-00005` | 816 | 42 |
| `NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB` | 74 | 9 |
| `NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-ctx16k` | 74 | 9 |
| `NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-ctx32k` | 74 | 9 |
| `NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-ctx8k` | 74 | 9 |
| `Qwen3-Coder-30B-A3B-Instruct` | 30 | 3 |
| `Qwen3.6-35B-A3B` | 35 | 3 |
| `Qwen3.6-35B-A3B-Q4_K_M.gguf` | 35 | 3 |
| `lmstudio-community-Qwen3-Next-80B-A3B-Instruct-MLX-4bit` | 80 | 3 |
| `mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit` | 30 | 3 |
| `mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit` | 30 | 3 |

### 4.3. Hạn chế của nhận diện MoE

- Extractor chỉ nhận diện MoE khi tên chứa token `A<number>B`. Model MoE không có token này sẽ bị xếp vào `dense` (ví dụ `gpt-oss-20b` được trích xuất 20B và xếp dense).
- Dataset không có cột kiến trúc để xác minh, nên không thể kết luận chắc chắn đâu là MoE nếu tên không ghi.
- 23 tên `unknown` bao gồm cả các model rất nhanh trong dữ liệu (ví dụ `DeepSeek-Coder-V2-Lite-Instruct` ~251 TPS, `Sage-A1.5-*` ~213 TPS). Điều này quan trọng cho biến thể C: imputation median 30B sẽ gán sai kích thước cho nhóm này.

---

## 5. Thiết kế ba biến thể A/B/C

### 5.1. Feature set

| Biến thể | Categorical | Numeric | Ghi chú |
|---|---|---|---|
| **A — Baseline** | `model`, `hardware`, `runtime`, `workload`, `quant` | `prompt_tokens`, `batch_size` | Đúng A1 hiện tại |
| **B — Augmented** | `model`, `hardware`, `runtime`, `workload`, `quant` | `model_params_b`, `prompt_tokens`, `batch_size` | Giữ `model`, thêm params |
| **C — Parameter-based** | `hardware`, `runtime`, `workload`, `quant` | `model_params_b`, `prompt_tokens`, `batch_size` | Bỏ `model`, thay bằng params |

### 5.2. Preprocessing

- `OneHotEncoder(handle_unknown="ignore")` cho categorical.
- `SimpleImputer(strategy="median")` cho `model_params_b`, **chỉ fit trên training set**.
- Median học được trên training seed 42: **30,0B**.
- Không xóa row nào; 145 rows thiếu `model_params_b` được impute.
- Không dùng `model_size`, không dùng thông tin ngoài dataset.

### 5.3. Protocol

| Protocol | Group | test_size | random_state | Mục đích |
|---|---|---:|---|---|
| `run_id` holdout | `run_id` | 0,2 | 42 (chính), 42–61 (robustness) | Tái lập đánh giá A1 tiêu chuẩn |
| `model` holdout | `model` | 0,2 | 42 (chính), 42–61 (robustness) | Test model chưa từng thấy |

- Cùng một `train_idx`/`test_idx` cho cả A, B, C trong mỗi protocol/seed.
- Kiểm tra seed 42:
  - `run_id` protocol: 595 train / 156 test; 268 / 67 `run_id`; giao `run_id` = 0.
  - `model` protocol: 604 train / 147 test; 84 / 22 model; giao `model` = 0; 22/22 test model chưa từng thấy.
- Không có target missing (751/751 rows có `decode_tps`).

### 5.4. Số liệu missing của `model_params_b`

| Protocol | Seed | Train rows | Test rows | Train thiếu | Test thiếu |
|---|---:|---:|---:|---:|---:|
| `run_id` | 42 | 595 | 156 | 112 | 33 |
| `model` | 42 | 604 | 147 | 99 | 46 |

---

## 6. Kết quả Holdout tiêu chuẩn (`GroupShuffleSplit` theo `run_id`)

### 6.1. Seed 42 — tái lập baseline

| Variant | n train | n test | MAE | RMSE | R² | Negative | Pred min | Pred max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 595 | 156 | **24,6859** | **40,3109** | **0,7108** | 0 | 9,79 | 329,84 |
| B | 595 | 156 | 24,7292 | 40,2485 | 0,7117 | 0 | 8,77 | 330,28 |
| C | 595 | 156 | 59,5196 | 72,4177 | 0,0665 | 0 | 34,73 | 158,48 |

Baseline A tái lập **chính xác** giá trị kỳ vọng:

| Metric | Kỳ vọng | Tái lập | Khớp |
|---|---:|---:|---|
| MAE | 24,6859 | 24,6859 | Có |
| RMSE | 40,3109 | 40,3109 | Có |
| R² | 0,7108 | 0,7108 | Có |

### 6.2. 20 seeds (42–61) — trung bình ± độ lệch chuẩn

| Variant | n seeds | MAE mean ± std | RMSE mean ± std | R² mean ± std | Negative total |
|---|---:|---:|---:|---:|---:|
| A | 20 | **27,7618 ± 5,6332** | 51,9526 ± 16,1425 | 0,5871 ± 0,1751 | 10 |
| B | 20 | **27,7456 ± 5,6976** | 51,7084 ± 16,0284 | 0,5913 ± 0,1728 | 10 |
| C | 20 | **60,3614 ± 6,4044** | 79,5998 ± 11,0375 | 0,0451 ± 0,1009 | 2 |

Test n trung bình: 153,2 (min 132, max 168). Số model chưa thấy trong test trung bình: 11,6 (7–17).

### 6.3. So sánh cặp theo từng seed

| So sánh | ΔMAE mean ± std | Số seed tốt hơn | Diễn giải |
|---|---:|---:|---|
| B − A | **−0,016 ± 0,307** | 9/20 | Gần như bằng 0; không có cải thiện thực chất |
| C − A | **+32,600 ± 4,650** | 0/20 | C tệ hơn ở **mọi** seed |
| C − B | +32,616 ± 4,641 | 0/20 | C tệ hơn B ở mọi seed |

### 6.4. Hệ số và permutation importance (seed 42, `run_id`)

| Variant | Hệ số `model_params_b` (TPS/B) | Intercept | Imputer median |
|---|---:|---:|---:|
| B | −0,245140 | 104,9830 | 30,0 |
| C | −0,183798 | 106,2774 | 30,0 |

Diễn giải thận trọng: trong mô hình tuyến tính, tăng 10B tổng tham số tương quan với giảm khoảng 2,45 TPS (B) hoặc 1,84 TPS (C), khi giữ các feature đã mã hóa khác cố định. Đây là **liên hệ thống kê trong mô hình**, không phải quan hệ nhân quả.

Permutation importance (tăng MAE khi hoán vị feature, 10 lần lặp):

| Protocol | Variant | Feature | ΔMAE mean ± std |
|---|---|---|---:|
| `run_id` | B | `model_params_b` | +0,4914 ± 0,1630 |
| `run_id` | B | `model` | **+50,7981 ± 1,8270** |
| `run_id` | C | `model_params_b` | −0,2348 ± 0,1227 |
| `model` | B | `model_params_b` | +0,7307 ± 0,1254 |
| `model` | B | `model` | 0,0000 ± 0,0000 (mọi test model đều unseen → OHE toàn 0) |
| `model` | C | `model_params_b` | +0,5798 ± 0,1084 |

Kết luận: `model` mang thông tin dự đoán lớn hơn `model_params_b` khoảng hai bậc độ lớn. Không thể nói `model_params_b` là yếu tố quan trọng nhất.

---

## 7. Kết quả Unseen-model Generalization

### 7.1. Seed 42

| Variant | n train | n test | n unseen models | MAE | RMSE | R² | Negative | Pred min | Pred max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 604 | 147 | 22/22 | 59,4505 | 68,5815 | −0,2431 | 0 | 22,19 | 167,71 |
| B | 604 | 147 | 22/22 | 60,2921 | 69,2526 | −0,2675 | 0 | 23,43 | 174,90 |
| C | 604 | 147 | 22/22 | **48,9158** | **56,6142** | **0,1529** | 0 | 13,97 | 155,59 |

Ở seed 42, C tốt hơn A khoảng 10,5 MAE. Tuy nhiên, đây là **một seed đơn lẻ** và không đại diện cho toàn bộ phân phối.

### 7.2. 20 seeds (42–61)

| Variant | n seeds | MAE mean ± std | RMSE mean ± std | R² mean ± std | Negative total |
|---|---:|---:|---:|---:|---:|
| A | 20 | **64,7528 ± 12,1392** | 81,1975 ± 15,2993 | 0,0241 ± 0,2628 | 8 |
| B | 20 | **67,9885 ± 15,6172** | 92,3474 ± 39,3785 | −0,5293 ± 1,8402 | 17 |
| C | 20 | **69,9196 ± 13,3525** | 93,2484 ± 28,3933 | −0,3851 ± 1,0463 | 43 |

Test n trung bình: 152,8 (min 102, max 222). Số model chưa thấy: 22 ở mọi seed.

### 7.3. So sánh cặp theo từng seed

| So sánh | ΔMAE mean ± std | Số seed tốt hơn | Diễn giải |
|---|---:|---:|---|
| B − A | **+3,236 ± 9,154** | 5/20 | B tệ hơn trung bình; không ổn định |
| C − A | **+5,167 ± 11,415** | 7/20 | C tệ hơn trung bình; seed 42 chỉ là ngoại lệ |
| C − B | +1,931 ± 12,007 | 9/20 | Không có bên nào vượt trội rõ |

### 7.4. Phân tầng theo khả năng trích xuất params của test model

| Subset | n rows/test (mean) | A MAE | B MAE | C MAE | B tốt hơn A | C tốt hơn A |
|---|---:|---:|---:|---:|---:|---:|
| Tất cả | 152,8 | 64,753 | 67,988 | 69,920 | 5/20 | 7/20 |
| Test model có params | 124,85 | 65,145 | 68,895 | 69,838 | 4/20 | 9/20 |
| Test model thiếu params | 27,95 | 63,670 | 63,521 | 70,152 | 14/20 | 4/20 |

Ngay cả trên subset test model **có** `model_params_b`, C vẫn không vượt A ổn định (chỉ 9/20 seed). Điều này cho thấy lợi thế seed 42 của C không đến từ việc "hiểu kích thước model" một cách bền vững.

### 7.5. Độ nhạy với active parameters (MoE)

Thay `model_params_b` (tổng) bằng `model_params_active_b` (active; dense = total) cho B và C:

| Protocol | Variant | MAE mean ± std | Negative total |
|---|---|---:|---:|
| `run_id` | B (active) | 28,4361 ± 6,3216 | 33 |
| `run_id` | C (active) | 43,8382 ± 7,4484 | 30 |
| `model` | B (active) | **54,4450 ± 10,9187** | 57 |
| `model` | C (active) | **50,9784 ± 12,0548** | 83 |

So sánh với A (không đổi): `run_id` A = 27,7618; `model` A = 64,7528.

- Trên unseen-model, active params cải thiện MAE đáng kể so với total params (B: 67,99 → 54,45; C: 69,92 → 50,98) và cả hai đều tốt hơn A về MAE trung bình.
- Nhưng số dự đoán âm tăng mạnh (B: 17 → 57; C: 43 → 83), và trên `run_id` C vẫn tệ hơn A rất nhiều (43,84 so với 27,76).
- Đây là tín hiệu đáng nghiên cứu tiếp, **không** phải bằng chứng đủ để triển khai.

---

## 8. So sánh dự đoán âm

### 8.1. Tổng hợp 20 seeds

| Protocol | Variant | Negative total | Negative mean/split | Non-finite | Inference errors |
|---|---|---:|---:|---:|---:|
| `run_id` | A | 10 | 0,50 | 0 | 0 |
| `run_id` | B | 10 | 0,50 | 0 | 0 |
| `run_id` | C | **2** | 0,10 | 0 | 0 |
| `model` | A | 8 | 0,40 | 0 | 0 |
| `model` | B | 17 | 0,85 | 0 | 0 |
| `model` | C | **43** | 2,15 | 0 | 0 |

### 8.2. Active-params sensitivity

| Protocol | Variant | Negative total |
|---|---|---:|
| `run_id` | B (active) | 33 |
| `run_id` | C (active) | 30 |
| `model` | B (active) | 57 |
| `model` | C (active) | 83 |

### 8.3. Nhận xét

- Seed 42: cả A, B, C đều có **0** dự đoán âm trên cả hai protocol.
- Không có lỗi inference và không có dự đoán NaN/Inf trong toàn bộ 120 lần fit multi-seed và 80 lần fit active-params.
- Thêm `model_params_b` (B) **không** giảm dự đoán âm; trên unseen-model còn tăng (8 → 17).
- C giảm dự đoán âm trên `run_id` (10 → 2) nhưng tăng mạnh trên unseen-model (8 → 43). Vì vậy **không thể** kết luận "thêm params giúp giảm dự đoán âm".
- Active params cải thiện MAE unseen-model nhưng làm tăng dự đoán âm ở cả hai protocol. Đây là trade-off quan trọng, không phải cải thiện thuần túy.

---

## 9. Phân tích và trả lời câu hỏi nghiên cứu

### 9.1. Bằng chứng định lượng bổ sung

**Tương quan (chỉ mô tả, không nhân quả):**

| Tập | n | Pearson | Spearman |
|---|---:|---:|---:|
| Rows có params | 606 | −0,1294 | −0,2623 |
| Một dòng / model | 83 | −0,1789 | −0,4900 |
| Một dòng / model, log-log | 83 | −0,5509 | — |
| Dense only | 66 | −0,6727 | — |
| MoE only | 17 | −0,4067 | — |

**Khả năng giải thích phương sai giữa các model:**

| Tập | R² của mean TPS ~ params (linear) | R² (log-log) |
|---|---:|---:|
| Tất cả model | 0,032 | 0,3034 |
| Dense only | 0,4526 | 0,6315 |

**Phân rã phương sai `decode_tps`:**

| Tập | Between-model | Within-model |
|---|---:|---:|
| 106 model names, 751 rows | **83,00%** | 17,00% |
| Model có params, 606 rows | **93,68%** | 6,32% |

Diễn giải: phần lớn biến thiên `decode_tps` nằm **giữa các model**, không phải trong cùng một model. `model_params_b` là thuộc tính cấp model (không đổi trong cùng một tên model), nên nó chỉ có thể giải thích một phần của 83% đó — và thực tế chỉ giải thích 3,2% (linear) hoặc 30,3% (log-log) phương sai giữa model. Định danh `model` giải thích được toàn bộ phần between-model theo thiết kế one-hot.

**Dải TPS rộng ở cùng quy mô 25–35B:**

| Nhóm | Khoảng TPS trung bình | Ví dụ |
|---|---:|---|
| Dense 30B | 20,78 – 80,56 | `Muse-Glimmer-30B-ctx16k` 20,78; `Qwen3.8-27B-mtp-on` 80,56 |
| MoE 30–35B | 104,01 – 235,62 | `mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit` 104,01; `Qwen3-Coder-30B-A3B-Instruct` 235,62 |
| Toàn dải 25–35B | **7,05 – 235,62** | `mlx-community/Qwen2.5-32B-Instruct-4bit` 7,05 |

Cùng khoảng 30B nhưng TPS chênh tới ~33 lần. `model_params_b` một mình không thể phân biệt các trường hợp này; cần thêm kiến trúc, active params, quantization, runtime, hardware.

**Nhóm thiếu params có tốc độ cao hơn:**

| Nhóm | n rows | Mean TPS | Median TPS |
|---|---:|---:|---:|
| Có `model_params_b` | 606 | 95,67 | 69,46 |
| Thiếu `model_params_b` | 145 | **142,98** | 133,87 |

Ví dụ trong nhóm thiếu: `DeepSeek-Coder-V2-Lite-Instruct` ~251 TPS, `Sage-A1.5-*` ~213 TPS. Việc impute median 30B cho các model này là một nguồn sai số hệ thống của biến thể C.

### 9.2. Trả lời 7 câu hỏi

**Câu 1 — Có thể trích xuất số lượng tham số từ tên model một cách đáng tin cậy không?**

Có, **với các tên chứa token kích thước rõ ràng**: 83/106 model (78,30%), 606/751 rows (80,69%), 25/25 synthetic test pass, 5/5 model có `model_size` khớp chính xác. Tuy nhiên **không đầy đủ**: 23/106 tên không có token, 145/751 rows thiếu, và nhận diện MoE chỉ dựa vào `A<number>B` nên bỏ sót MoE không ghi active params. Không nên dùng tên model làm nguồn duy nhất cho metadata kiến trúc.

**Câu 2 — Bổ sung `model_params_b` có cải thiện holdout chuẩn không?**

**Không.** B gần như trùng A: MAE 24,7292 so với 24,6859; ΔMAE trung bình −0,016 ± 0,307; chỉ 9/20 seed tốt hơn. R² nhích nhẹ (0,7117 so với 0,7108) nhưng không đủ để coi là cải thiện thực chất. C tệ hơn rõ rệt: MAE 59,5196, 0/20 seed tốt hơn.

**Câu 3 — Có cải thiện tổng quát hóa sang model chưa từng thấy không?**

**Không ổn định.** B tệ hơn trung bình (+3,236 MAE; 5/20 seed tốt hơn). C tệ hơn trung bình (+5,167 MAE; 7/20 seed tốt hơn), dù seed 42 C tốt hơn. Phân tầng cho thấy ngay cả khi test model có params, C chỉ tốt hơn A ở 9/20 seed. Không có bằng chứng đủ mạnh để kết luận B hoặc C cải thiện generalization.

**Câu 4 — Thay danh tính model bằng số lượng tham số có giúp giảm ghi nhớ tên model không?**

Có giảm việc "ghi nhớ tên" theo nghĩa cơ học (C bỏ hẳn `model` khỏi feature set), nhưng **không** dẫn đến tổng quát hóa tốt hơn. Ngược lại, C mất thông tin định danh model và holdout chuẩn sụp đổ (MAE 24,69 → 59,52). `model_params_b` là hằng số trong cùng một model, nên không thể giải thích 17% within-model variance (hoặc 6,32% ở nhóm có params). Định danh model vẫn là feature quan trọng nhất trong dữ liệu hiện tại.

**Câu 5 — Bổ sung số lượng tham số có làm giảm tỷ lệ dự đoán âm không?**

**Không nhất quán.** B: 10 âm trên `run_id` (bằng A) và 17 trên unseen-model (tăng so với A = 8). C: giảm còn 2 trên `run_id` nhưng tăng lên 43 trên unseen-model. Active params còn tệ hơn (B 57, C 83 trên unseen-model). Không có biến thể nào giảm âm một cách nhất quán trên cả hai protocol.

**Câu 6 — `model_params_b` mang thông tin bổ sung hay thay thế hoàn toàn được `model`?**

**Bổ sung rất ít; không thể thay thế.** Trong B, hệ số params có ý nghĩa thống kê trong mô hình (≈ −0,245 TPS/B) và permutation importance +0,49 MAE, nhưng `model` có permutation importance +50,80 MAE — lớn hơn khoảng 100 lần. Trong C, việc bỏ `model` làm MAE tăng 32,6. Vậy params chứa một ít tín hiệu bổ sung nhưng không thay thế được định danh model.

**Câu 7 — Những giới hạn nào còn tồn tại?**

- MoE: total params khác active params; `model_params_b` (total) không phản ánh chi phí suy luận thực tế. Active params cho tín hiệu tốt hơn trên unseen-model nhưng làm tăng dự đoán âm và vẫn không cứu được `run_id`.
- Nhận diện MoE không đầy đủ nếu tên không có `A<number>B` (ví dụ `gpt-oss-20b` bị xếp dense).
- 23 tên model không có thông tin kích thước; 145 rows phải impute median 30B, trong khi nhóm này có mean TPS 142,98 — imputation dễ sai hệ thống.
- `model_size` thiếu 84,55% và có 3 tên model chứa nhiều giá trị khác nhau; không dùng làm fallback.
- Tên model có thể gộp nhiều biến thể khác nhau (cùng tên nhưng khác kích thước/quantization).
- Dataset nhỏ: 106 model, 751 rows, 335 `run_id`; kết quả có phương sai cao giữa các seed.
- Mô hình tuyến tính: quan hệ giữa params và TPS có thể phi tuyến; log-log cải thiện R² giữa model (0,3034 so với 0,032) nhưng chưa được đưa vào A/B/C để giữ scope.
- Chỉ đánh giá bằng MAE/RMSE/R² và số dự đoán âm; chưa đánh giá calibration hay sai số theo nhóm hardware.
- Tương quan/permutation importance không phải nhân quả; không có thí nghiệm can thiệp (intervention) để xác định quan hệ nhân quả.

---

## 10. Giới hạn của nghiên cứu

1. **Không có metadata kiến trúc đáng tin cậy.** Extractor chỉ đọc tên; không xác minh được dense/MoE, active params, hay biến thể quantization.
2. **Imputation đơn giản.** Median 30B là lựa chọn tối thiểu để không xóa dòng; chưa thử missing indicator, model-family median, hay imputation theo nhóm. Điều này đặc biệt bất lợi cho C.
3. **Không dùng `model_size`.** Cột này thiếu 84,55%, không nhất quán theo tên model, và có thể là thông tin chỉ có trong benchmark (không có khi inference), nên không phù hợp làm feature production.
4. **Một mô hình tuyến tính.** Kết quả có thể khác với mô hình phi tuyến/cây; nghiên cứu này chỉ kiểm chứng giả thuyết trong cùng họ mô hình A1.
5. **Phương sai seed cao.** Unseen-model MAE std ~12–16; kết luận dựa trên trung bình 20 seed và so sánh cặp, không dựa trên một split đẹp.
6. **Không đánh giá trên dữ liệu ngoài dataset.** Mọi kết luận chỉ đúng trong phân phối 751 rows hiện có.
7. **Không có kiểm định thống kê chính thức.** Báo cáo dùng mean/std và số seed thắng; chưa dùng paired test/Bootstrap CI. Vì vậy không nên suy diễn quá mức về "ý nghĩa thống kê".
8. **Không thay đổi production.** Mọi kết quả là nghiên cứu; artifact, app và inference giữ nguyên.

---

## 11. Khuyến nghị cho CDK

### 11.1. Quyết định

| Hạng mục | Khuyến nghị | Lý do |
|---|---|---|
| A1 production | **Giữ nguyên** | Tái lập đúng baseline; không có biến thể nào vượt ổn định |
| Biến thể B | **Không triển khai** | Không cải thiện holdout chuẩn; thêm missingness/phức tạp |
| Biến thể C | **Không triển khai** | Holdout chuẩn tệ hơn ~2,4 lần MAE; unseen-model không ổn định; tăng dự đoán âm |
| Active params | **Chỉ nghiên cứu tiếp** | Có tín hiệu trên unseen-model MAE nhưng tăng dự đoán âm và vẫn tệ trên `run_id` |
| Guard dự đoán âm | **Giữ nguyên** | Nghiên cứu này không thay đổi hành vi UI/guard |

### 11.2. Có nên nghiên cứu tiếp B?

Không ưu tiên. B gần như trùng A; chi phí thêm feature, imputation và giải thích không mang lại lợi ích đo được. Nếu vẫn muốn theo đuổi, cần một nguồn metadata kích thước đầy đủ hơn và một thí nghiệm riêng có kiểm định thống kê.

### 11.3. Có nên nghiên cứu tiếp C?

Không ở dạng hiện tại. C chỉ có lợi thế ở một số seed unseen-model, không ổn định, và làm mất thông tin định danh model. Hướng đáng nghiên cứu hơn là **C + active params + metadata kiến trúc đầy đủ + missing indicator**, nhưng đây là nghiên cứu mới, cần được phê duyệt riêng, không được tự ý triển khai.

### 11.4. Có đủ bằng chứng để thay đổi feature engineering của CDK chưa?

**Chưa.** Tiêu chí tối thiểu nên là: cải thiện ổn định trên cả `run_id` holdout và unseen-model, không tăng dự đoán âm, và có thể tái lập qua nhiều seed. Không biến thể nào đáp ứng đồng thời các tiêu chí này.

### 11.5. Việc cần làm nếu tiếp tục

1. Xây dựng metadata model có nguồn gốc rõ ràng (architecture, total/active params, quantization) thay vì chỉ parse tên.
2. Thử missing indicator và imputation theo nhóm thay vì median toàn cục.
3. Đánh giá thêm mô hình phi tuyến (ví dụ cây) trong một nghiên cứu riêng, không thay A1.
4. Dùng paired test/Bootstrap CI cho so sánh A/B/C.
5. Giữ nguyên guard dự đoán âm và các disclaimer hiện có.

---

## 12. Phụ lục — Tái lập

### 12.1. Cấu trúc thực nghiệm

- Dataset đọc trực tiếp từ `data/raw/llm-speed-runs-2026-10-22.6.csv`; không ghi lại.
- `quant`: `NaN` → `"Unknown"`.
- Target: `decode_tps`; 751/751 rows có target.
- Không xóa row nào cho `model_params_b`.
- Tất cả pipeline được fit **chỉ trên training set** của từng split.

### 12.2. Code trích xuất (rút gọn, đủ để tái lập)

```python
import re

TOTAL_RE = re.compile(r"(?<![A-Za-z0-9.])(\d+(?:\.\d+)?)\s*[Bb]\b")
ACTIVE_RE = re.compile(r"(?<![A-Za-z0-9])[Aa](\d+(?:\.\d+)?)\s*[Bb]\b")
MOE_X_RE = re.compile(r"(?<![A-Za-z0-9])(\d+(?:\.\d+)?)\s*[xX]\s*(\d+(?:\.\d+)?)\s*[Bb]\b")


def extract_model_params(name: str) -> dict:
    text = str(name)
    x_matches = MOE_X_RE.findall(text)
    total_vals = sorted({float(x) for x in TOTAL_RE.findall(text)})
    active_vals = sorted({float(x) for x in ACTIVE_RE.findall(text)})

    if x_matches and not total_vals:
        return {"model": text, "status": "ambiguous", "model_params_b": None,
                "active_params_b": None, "is_moe": True}
    if len(total_vals) > 1:
        return {"model": text, "status": "ambiguous", "model_params_b": None,
                "active_params_b": None, "is_moe": bool(active_vals)}
    if len(total_vals) == 1:
        total = total_vals[0]
        if active_vals:
            if len(active_vals) > 1:
                return {"model": text, "status": "ambiguous", "model_params_b": None,
                        "active_params_b": None, "is_moe": True}
            return {"model": text, "status": "moe_explicit_total",
                    "model_params_b": total, "active_params_b": active_vals[0], "is_moe": True}
        return {"model": text, "status": "dense", "model_params_b": total,
                "active_params_b": total, "is_moe": False}
    if active_vals:
        return {"model": text, "status": "ambiguous", "model_params_b": None,
                "active_params_b": active_vals[0] if len(active_vals) == 1 else None,
                "is_moe": True}
    return {"model": text, "status": "unknown", "model_params_b": None,
            "active_params_b": None, "is_moe": False}
```

### 12.3. Pipeline A/B/C

```python
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

CAT_A = ["model", "hardware", "runtime", "workload", "quant"]
CAT_C = ["hardware", "runtime", "workload", "quant"]
NUM = ["prompt_tokens", "batch_size"]

def features(variant):
    if variant == "A":
        return CAT_A + NUM
    if variant == "B":
        return CAT_A + ["model_params_b"] + NUM
    if variant == "C":
        return CAT_C + ["model_params_b"] + NUM

def build_pipeline(variant):
    cat = CAT_A if variant in ("A", "B") else CAT_C
    parts = [("cat", OneHotEncoder(handle_unknown="ignore"), cat)]
    if variant in ("B", "C"):
        parts.append((
            "num_params",
            Pipeline([("imputer", SimpleImputer(strategy="median"))]),
            ["model_params_b"],
        ))
    parts.append(("num", "passthrough", NUM))
    return Pipeline([
        ("pre", ColumnTransformer(parts, remainder="drop")),
        ("reg", LinearRegression()),
    ])
```

### 12.4. Split

```python
from sklearn.model_selection import GroupShuffleSplit

def make_split(df, protocol, seed):
    groups = df["run_id"] if protocol == "run_id" else df["model"]
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    return next(gss.split(df, df["decode_tps"], groups=groups))
```

- Seed chính: 42.
- Robustness: 42–61 (20 seeds).
- Mỗi split được tạo **một lần**, dùng chung cho A/B/C.

### 12.5. Artifact trung gian

Các file CSV/JSON trung gian của phiên nghiên cứu nằm ngoài repo, tại:

`C:\Users\Admin\AppData\Local\Temp\opencode\cdk_params\`

Bao gồm:

- `extraction_audit.csv`, `extraction_synthetic_tests.csv`, `model_size_comparison.csv`
- `dataset_with_params.csv`
- `primary_seed42.csv`, `multiseed_results.csv`, `multiseed_aggregate.csv`
- `permutation_importance_seed42.csv`
- `moe_active_sensitivity.csv`, `moe_active_aggregate.csv`
- `multiseed_unseen_stratified.csv`, `multiseed_unseen_stratified_agg.csv`
- `evaluation_results.json`

Không có file nào trong số này được stage/commit. Nếu cần lưu lâu dài, chủ dự án quyết định có đưa vào repo hay không.

---

## 13. Tuân thủ phạm vi

### 13.1. File thay đổi

| File | Trạng thái |
|---|---|
| `reports/model_parameter_feature_investigation.md` | **Mới** (báo cáo này) |
| `app.py` | Không thay đổi bởi nghiên cứu này (đang có thay đổi từ patch guard đã được duyệt trước đó) |
| `src/inference.py` | Không thay đổi |
| `models/cdk_a1_deployment.joblib` | Không thay đổi |
| `data/raw/llm-speed-runs-2026-10-22.6.csv` | Không thay đổi |
| `notebooks/02_data_analysis.ipynb`, `notebooks/03_model.ipynb` | Không thay đổi |
| `requirements.txt` | Không thay đổi |

### 13.2. Hash kiểm chứng

| File | SHA-256 |
|---|---|
| `data/raw/llm-speed-runs-2026-10-22.6.csv` | `0b7a57d6f2a8ef768f161df143d665c2a3723dfb21b426a289ea4e062f522eb2` |
| `models/cdk_a1_deployment.joblib` | `b112a9bada0383e134c32a055f82e0c2b40e1a26c50d0a5bd55aa8bbd70d192d` |
| `src/inference.py` | `cf8c4d180c9907263762d19fca964405758f2736862ef0b5f3dde0da5a568a60` |
| `notebooks/02_data_analysis.ipynb` | `a8e54c6e09e315d470d0e51ec955e2439cb7b41b572d6ace4921b86b6b7782b6` |
| `notebooks/03_model.ipynb` | `2a162c1af80eebe2da86ca5c933cf3bb6ecd5f3986c224f606712d705f5d00c5` |

### 13.3. Git

- Không commit, không push, không stage.
- `reports/` vẫn là untracked như trước nghiên cứu.
- Không có thay đổi nào ngoài file báo cáo này.

---

## 14. Kết luận cuối

Nghiên cứu **không tìm thấy bằng chứng đủ mạnh** để thêm `model_params_b` vào CDK. Việc trích xuất tham số từ tên model là khả thi và hữu ích cho phân tích, nhưng:

- Không cải thiện holdout chuẩn (B gần như trung tính; C tệ hơn rõ rệt).
- Không cải thiện tổng quát hóa unseen-model một cách ổn định.
- Không giảm dự đoán âm một cách nhất quán.
- Không thay thế được `model`; định danh model vẫn là feature quan trọng nhất.

**Giữ A1 làm production model. Không triển khai B hoặc C.** Hướng `active_params_b` chỉ nên được xem là giả thuyết cho nghiên cứu tiếp theo, với metadata kiến trúc đầy đủ hơn và một protocol đánh giá nghiêm ngặt hơn.

---

*Hết báo cáo. Chờ chủ dự án review trước khi có bất kỳ thay đổi nào tiếp theo.*
