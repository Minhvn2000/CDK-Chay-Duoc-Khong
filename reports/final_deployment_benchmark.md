# CDK — Final Deployment Benchmark & Prediction Audit

**Ngày audit:** 2026-11-07  
**Phạm vi:** `models/cdk_a1_deployment.joblib`, `src/inference.py`, `app.py`, `data/raw/llm-speed-runs-2026-10-22.6.csv`, protocol đánh giá A1/A2 hiện có.  
**Nguyên tắc thực hiện:** Chỉ đọc, chạy kiểm thử và tạo báo cáo. **Không retrain/chỉnh sửa artifact deployment**, không đổi dataset/feature set/hyperparameters, không sửa `app.py`/`src/inference.py`, không sửa raw data. Các kết quả dự đoán âm **không được dùng như TPS hợp lệ**.

---

## 1. Executive Summary

### 1.1 Kết luận điều hành

| Hạng mục | Kết luận |
|---|---|
| Artifact deployment | **PASS** — load thành công, đúng A1 deployment pipeline, schema/feature order khớp `src/inference.py`. |
| Inference kỹ thuật | **PASS** — 2.190 lượt kiểm thử, **0 lỗi inference**, không có NaN/Inf. |
| Holdout A1 (theo `run_id`) | **PASS** — tái lập đúng số liệu đã công bố: LinearRegression MAE **24.6859**, RMSE **40.3109**, R² **0.7108**; Dummy MAE **52.6869**, RMSE **82.2901**, R² **−0.2053**. |
| Dự đoán trên cấu hình đã quan sát | **WARNING** — 2/751 dự đoán âm (0,27%); in-sample MAE 7.8251, R² 0.9661. |
| Dự đoán trên cấu hình chưa quan sát | **FAIL về tính hợp lệ** — 46/200 (23,0%) âm khi `hardware` chưa thấy; 116/500 (23,2%) âm khi tổ hợp feature chưa thấy; 120/500 (24,0%) âm với input hợp lệ ngẫu nhiên. |
| Boundary input | **PASS** — 0/39 dự đoán âm với prompt min/median/max × 13 giá trị `quant`. |
| Sẵn sàng demo | **WARNING có điều kiện** — đủ điều kiện **demo có kiểm soát**; **không đủ điều kiện demo tự do** hoặc tuyên bố dự đoán cho mọi cấu hình. |

### 1.2 Con số then chốt

- Tổng lượt kiểm thử: **2.190**; lỗi inference: **0**; dự đoán âm: **284** (12,97% toàn bộ suite, nhưng phân bố rất không đều giữa các nhóm).
- Nhóm cấu hình đã quan sát (751 hàng): **2 âm / 751** = **0,27%**; 749 dương; 0 bằng 0.
- Nhóm `hardware` chưa thấy (200): **46 âm / 200** = **23,0%**.
- Nhóm tổ hợp feature chưa thấy (500): **116 âm / 500** = **23,2%**.
- Nhóm input hợp lệ ngẫu nhiên (500): **120 âm / 500** = **24,0%**.
- Nhóm `model` chưa thấy (200): **0 âm**, nhưng mean predicted TPS **121,353** so với nhóm known **104,806** — lệch hướng do `OneHotEncoder` zero hoá category chưa thấy.
- Boundary (39): **0 âm**; min **11,432**, max **135,513** TPS.

### 1.3 Verdict

**CDK đủ điều kiện để demo trong buổi thuyết trình nếu demo có kiểm soát**, cụ thể:
- chỉ dùng cấu hình **đã có trong dataset** và đã kiểm tra trước cho ra dự đoán dương;
- không chọn ngẫu nhiên các selectbox;
- giải thích rõ negative prediction là **không hợp lệ**, không phải TPS;
- không tuyên bố mô hình tổng quát hoá cho `model`/`hardware`/tổ hợp chưa thấy.

Nếu demo cho phép người xem chọn tự do mọi tổ hợp trong UI → **không đạt** vì xác suất gặp dự đoán âm rất cao (≈23–24% trong các nhóm chưa thấy).

---

## 2. Environment and Artifact Verification

### 2.1 Môi trường chạy

| Thành phần | Phiên bản |
|---|---|
| Python | 3.11.9 |
| streamlit | 1.59.0 |
| pandas | 3.0.3 |
| numpy | 2.4.6 |
| scikit-learn | 1.9.0 |
| joblib | 1.5.3 |

Các phiên bản khớp với `requirements.txt` và môi trường đã tạo artifact.

### 2.2 Artifact

| Thuộc tính | Giá trị |
|---|---|
| Đường dẫn | `models/cdk_a1_deployment.joblib` |
| Kích thước | 8.948 bytes |
| SHA-256 | `b112a9bada0383e134c32a055f82e0c2b40e1a26c50d0a5bd55aa8bbd70d192d` |
| Load | Thành công, **không có warning** `InconsistentVersionWarning` |
| Kiểu | `dict` chứa key `pipeline` |
| `artifact_type` | `deployment` |
| `model_name` | `cdk_a1_deployment` |
| `target` | `decode_tps` |
| `rows_fit` (metadata) | **751** |
| Pipeline steps | `pre` = `ColumnTransformer`; `reg` = `LinearRegression` |
| Preprocessor | `cat` = `OneHotEncoder(handle_unknown="ignore")` cho 5 categorical; `num` = passthrough cho 2 numeric |
| Số category mỗi feature | model **106**, hardware **18**, runtime **3**, workload **5**, quant **13** (12 giá trị + `Unknown`) |
| Số hệ số | **147** |
| Intercept | **79,306136** |
| `n_features_in_` | **7** |
| Feature list | `["model","hardware","runtime","workload","quant","prompt_tokens","batch_size"]` |
| Khớp `src.inference.FEATURES` | **Có** |
| `_sklearn_version` lưu trong artifact | Không có trường này |

**Đánh giá:** Artifact đúng là A1 deployment pipeline: categorical one-hot + numeric passthrough + `LinearRegression`, fit trên toàn bộ dataset theo deployment policy. Feature names/order khớp `src/inference.py`.

**Về “đúng 751 observations”:** Metadata ghi `rows_fit = 751`; số category OHE khớp chính xác với toàn bộ dataset (106/18/3/5/13); mean prediction in-sample bằng mean actual (104,805739) và residual mean ≈ 0 — nhất quán với OLS fit trên toàn bộ 751 hàng. **Không thể tái đếm độc lập chính xác số hàng nếu không refit**, mà refit bị cấm trong audit này. Vì vậy đây là **PASS dựa trên metadata + kiểm tra nhất quán**, không phải bằng chứng refit độc lập.

### 2.3 Dataset tham chiếu

| Thuộc tính | Giá trị |
|---|---|
| Số hàng | 751 |
| Số cột | 18 |
| `run_id` unique | 335 |
| `model` unique | 106 |
| `hardware` unique | 18 |
| `runtime` | llama.cpp, mlx, ollama |
| `workload` | agent-trace, chat-long, chat-short, concurrent-decode, long-context-decay |
| `quant` | 12 giá trị + `Unknown` (273 hàng thiếu `quant`) |
| `prompt_tokens` min/median/max | 0 / 1024 / 23872 |
| `batch_size` | Luôn = 1 |
| Unique 5-tuple (model, hardware, runtime, workload, quant) | **544** |
| Unique 4-tuple (không gồm quant) | **531** |
| Tổng 5-tuple có thể có | **372.060** |
| Độ phủ 5-tuple quan sát | **0,146%** |

---

## 3. Benchmark Methodology

### 3.1 Cách chạy

- Dùng **canonical inference API** `src.inference.predict(payload, artifact)` cho mọi test.
- `batch_size = 1` cố định; `quant` thiếu → `Unknown` theo A1.
- Không tạo giá trị ngoài schema; không dùng `prefill_tps`, `ttft_ms`, `output_tokens`.
- Tất cả payload có đúng 7 feature.

### 3.2 Các nhóm kiểm thử

| Nhóm | Cách sinh | Số test |
|---|---|---|
| **A. Known configurations** | Toàn bộ 751 hàng của dataset (A1: `quant` thiếu → `Unknown`) | 751 |
| **B1. Unseen model** | 200 hàng mẫu; thay `model` bằng `__UNSEEN_MODEL__` | 200 |
| **B2. Unseen hardware** | 200 hàng mẫu; thay `hardware` bằng `__UNSEEN_HARDWARE__` | 200 |
| **B3. Unseen feature combination** | 500 tổ hợp 5-tuple từ các category **đã biết** nhưng **chưa từng xuất hiện** trong training; `prompt_tokens` lấy từ giá trị thực tế | 500 |
| **C. Boundary inputs** | Cấu hình mode; `prompt_tokens` ∈ {0, 1024, 23872} × 13 giá trị `quant` | 39 |
| **D. Randomized valid inputs** | 500 tổ hợp ngẫu nhiên từ các giá trị category thực tế; `prompt_tokens` lấy từ giá trị thực tế; `batch_size=1` | 500 |

Seed cố định: `numpy.random.default_rng(42)`.

### 3.3 Phân tích negative prediction

Với mỗi dự đoán âm, ghi lại:
- full payload;
- predicted TPS;
- tuple có xuất hiện trong training không (5-tuple và 4-tuple);
- category `model`/`hardware`/`runtime`/`workload`/`quant` có seen không;
- tần suất category trong training;
- percentile của `prompt_tokens`;
- phân rã tuyến tính theo từng feature gốc: `intercept + Σ(transformed_value × coefficient)`.

**Lưu ý collinearity:** `OneHotEncoder` dùng đủ mọi level (`drop=None`) + `LinearRegression` có intercept → ma trận thiết kế suy biến. Phân rã contribution là **một cách phân rã hợp lệ của linear predictor**, nhưng **không duy nhất** và không nên diễn giải từng hệ số riêng lẻ như quan hệ nhân quả.

### 3.4 Tái lập holdout A1/A2

Đúng protocol hiện có:
- `GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)`;
- group = `run_id`;
- A1: `quant` thiếu → `Unknown`, 751 hàng;
- A2: bỏ hàng thiếu `quant`, 478 hàng;
- fit `DummyRegressor(strategy="median")` và `LinearRegression` trên train, đánh giá trên test;
- kiểm tra intersection nhóm train/test = 0.

**Không dùng deployment artifact fit trên toàn dataset để tuyên bố holdout performance.** Deployment artifact chỉ dùng cho benchmark hành vi dự đoán.

---

## 4. Results by Test Group

### 4.1 Thống kê tổng hợp

| Nhóm | n | Lỗi | Âm | Bằng 0 | Dương | Min | Max | Mean | Median |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A. Known (751 hàng) | 751 | 0 | **2** | 0 | 749 | −16,7726 | 456,7148 | 104,8057 | 78,1585 |
| B1. Unseen model | 200 | 0 | **0** | 0 | 200 | 20,9930 | 409,2129 | 121,3533 | 119,9296 |
| B2. Unseen hardware | 200 | 0 | **46** | 0 | 154 | −21,7699 | 340,6670 | 62,7470 | 46,4917 |
| B3. Unseen combination | 500 | 0 | **116** | 0 | 384 | −164,3816 | 567,3014 | 74,2513 | 63,7248 |
| C. Boundary | 39 | 0 | **0** | 0 | 39 | 11,4324 | 135,5125 | 81,3384 | 84,5239 |
| D. Randomized valid | 500 | 0 | **120** | 0 | 380 | −211,4972 | 525,7067 | 78,4406 | 62,9683 |

**Tỷ lệ âm:**

| Nhóm | Tỷ lệ âm |
|---|---:|
| A. Known | 2/751 = **0,27%** |
| B1. Unseen model | 0/200 = **0,00%** |
| B2. Unseen hardware | 46/200 = **23,00%** |
| B3. Unseen combination | 116/500 = **23,20%** |
| C. Boundary | 0/39 = **0,00%** |
| D. Randomized valid | 120/500 = **24,00%** |

**Nhận xét:**
- Không có lỗi inference, không có NaN/Inf, không có dự đoán bằng 0.
- Dự đoán âm **chỉ xuất hiện ở cấu hình chưa từng thấy trong training**, ngoại trừ 2 trường hợp known (in-sample).
- B1 (unseen model) không âm nhưng **lệch mean +16,5 TPS** so với known (121,35 vs 104,81) — dấu hiệu bias do zero-hoá model effect, không phải an toàn.

### 4.2 Known configurations (in-sample, deployment fit)

| Metric | Giá trị |
|---|---:|
| n | 751 |
| MAE | 7,8251 |
| RMSE | 14,6501 |
| R² | 0,9661 |
| Residual mean | ≈ 0 |
| Residual std | 14,6501 |
| Residual min / max | −86,2504 / +83,3119 |
| Predicted min / max | −16,7726 / 456,7148 |
| Actual min / max | 3,76 / 466,95 |
| Predicted mean | 104,8057 |
| Actual mean | 104,8057 |

**Lưu ý:** Đây là **in-sample** (deployment artifact fit trên toàn bộ 751 hàng), **không phải holdout**. Không dùng các số này để tuyên bố generalization.

### 4.3 Boundary inputs

Cấu hình mode: `model = Qwen3.6-27B-Q4_K_M.gguf`, `hardware = RTX 5090 (32GB)`, `runtime = llama.cpp`, `workload = chat-short`, `quant = Q4_K_M`.

| `prompt_tokens` | n | Min | Max | Mean | Median |
|---|---:|---:|---:|---:|---:|
| 0 | 13 | 11,4324 | 115,9278 | 74,5301 | 72,0736 |
| 1024 | 13 | 12,2725 | 116,7679 | 75,3702 | 72,9137 |
| 23872 | 13 | 31,0171 | 135,5125 | 94,1149 | 91,6583 |

Theo `quant`:
- Thấp nhất: `UD-Q4_K_XL` (mean 18,241; range 11,432–31,017).
- Cao nhất: `Q8_0` (mean 122,736; range 115,928–135,513).
- `Unknown`: mean 76,620; range 69,812–89,397.

**Kết luận boundary:** Không có âm trong 39 test. Tuy nhiên đây chỉ là một cấu hình mode; không đại diện cho mọi tổ hợp.

### 4.4 Holdout A1/A2 (tái lập)

| Approach | Model | n train/test | Groups train/test | Intersection | MAE | RMSE | R² |
|---|---|---|---:|---:|---:|---:|---:|
| A1: quant → `Unknown` | DummyRegressor(median) | 595 / 156 | 268 / 67 | 0 | 52,6869 | 82,2901 | −0,2053 |
| A1: quant → `Unknown` | LinearRegression | 595 / 156 | 268 / 67 | 0 | **24,6859** | **40,3109** | **0,7108** |
| A2: drop quant NaN | DummyRegressor(median) | 378 / 100 | 112 / 28 | 0 | 58,6658 | 81,2513 | −0,5809 |
| A2: drop quant NaN | LinearRegression | 378 / 100 | 112 / 28 | 0 | 27,1261 | 44,7867 | 0,5197 |

Các số A1 khớp với kết quả đã công bố trước đó. A2 cũng khớp hướng so sánh (A1 tốt hơn A2).

---

## 5. Negative Prediction Case Studies

### 5.1 Tổng quan

Tổng số dự đoán âm: **284**.

| Nhóm | Số âm | 5-tuple có trong training | Model unseen | Hardware unseen |
|---|---:|---:|---:|---:|
| A. Known | 2 | **2** | 0 | 0 |
| B2. Unseen hardware | 46 | **0** | 0 | **46** |
| B3. Unseen combination | 116 | **0** | 0 | 0 |
| D. Randomized valid | 120 | **0** | 0 | 0 |

Ngoại trừ 2 trường hợp known, **toàn bộ 282 trường hợp âm còn lại đều có tuple chưa từng xuất hiện trong training**. Đây là bằng chứng trực tiếp cho vai trò của unseen combination/distribution shift.

### 5.2 Hai trường hợp known âm (in-sample)

**Case A-1**

| Trường | Giá trị |
|---|---|
| model | `mlx-community/Qwen3-32B-4bit` |
| hardware | `M3 Pro (18-core GPU) + 36GB unified` |
| runtime | `mlx` |
| workload | `chat-short` |
| quant | `Unknown` |
| prompt_tokens | 110 |
| batch_size | 1 |
| **predicted TPS** | **−16,7726** |
| **actual `decode_tps`** | **7,16** |
| run_id | `r_pnrrpcdqfo4` |
| tuple in training | Có |
| Contribution | intercept +79,306; model −59,791; hardware −35,795; runtime −4,841; workload +8,976; quant −4,718; prompt +0,090 |

**Case A-2**

| Trường | Giá trị |
|---|---|
| model | `mlx-community/Qwen2.5-32B-Instruct-4bit` |
| hardware | `M3 Pro (18-core GPU) + 36GB unified` |
| runtime | `mlx` |
| workload | `chat-short` |
| quant | `Unknown` |
| prompt_tokens | 131 |
| batch_size | 1 |
| **predicted TPS** | **−0,7657** |
| **actual `decode_tps`** | **7,05** |
| run_id | `r_f4x3xaan2ay` |
| tuple in training | Có |
| Contribution | intercept +79,306; model −43,801; hardware −35,795; runtime −4,841; workload +8,976; quant −4,718; prompt +0,107 |

**Diễn giải:** Đây là hạn chế của Linear Regression không có ràng buộc dương: với cấu hình MLX 32B trên M3 Pro, tổng các contribution âm vượt intercept, dù actual TPS vẫn dương (7,16 và 7,05). Đây là **in-sample**, nên không phải lỗi schema hay unseen category.

### 5.3 Unseen hardware (B2) — ví dụ điển hình

| model | hardware | runtime | workload | quant | prompt | pred | model contrib | hardware contrib | quant contrib |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `Qwen2.5-Coder-32B-Instruct` | `__UNSEEN_HARDWARE__` | llama.cpp | chat-long | Unknown | 0 | −21,7699 | −92,778 | **0,000** | −4,718 |
| `Qwen2.5-Coder-32B-Instruct` | `__UNSEEN_HARDWARE__` | llama.cpp | agent-trace | Unknown | 3362 | −18,1052 | −92,778 | **0,000** | −4,718 |
| `gemma-4-31B-it-Q4_K_M.gguf` | `__UNSEEN_HARDWARE__` | llama.cpp | chat-short | Unknown | 0 | −16,6787 | −97,241 | **0,000** | −4,718 |

**Điểm quan trọng:** `hardware` chưa thấy bị `OneHotEncoder(handle_unknown="ignore")` mã hoá thành **vector 0**, nên contribution của hardware = **0**, không phải “giá trị trung bình”. Khi model/quant có contribution âm mạnh, intercept +79,306 không đủ bù → dự đoán âm.

### 5.4 Unseen feature combination (B3) — ví dụ điển hình

| model | hardware | runtime | workload | quant | prompt | pred | model contrib | hardware contrib | runtime contrib | workload contrib | quant contrib | prompt contrib |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `Qwen2.5-Coder-32B-Instruct` | `RTX A4000 (16GB)` | llama.cpp | long-context-decay | UD-Q4_K_XL | 3188 | **−164,3816** | −92,778 | −71,004 | −3,001 | −16,422 | −63,098 | +2,615 |
| `mlx-community/Codestral-22B-v0.1-4bit` | `AMD64 ... (6c) + 15GB` | mlx | long-context-decay | Unknown | 242 | **−162,2768** | −60,606 | −155,195 | −4,841 | −16,422 | −4,718 | +0,199 |
| `Wicked-Tutu-31B-Final-ctx8k-kvq8` | `AMD64 ... (6c) + 15GB` | ollama | long-context-decay | Unknown | 3224 | **−160,0950** | −73,553 | −155,195 | +7,842 | −16,422 | −4,718 | +2,645 |

**Điểm quan trọng:** Tất cả category đều **đã thấy** trong training, nhưng **tổ hợp** thì chưa. Linear Regression cộng các hệ số của một tổ hợp chưa quan sát, tạo ra giá trị ngoài miền target. Đây là **extrapolation theo tổ hợp**, không phải unseen category.

### 5.5 Randomized valid inputs (D) — ví dụ điển hình

120/500 trường hợp âm; **cả 120 đều có 5-tuple chưa từng xuất hiện trong training**. Các feature đóng góp âm lớn nhất thường là `model` và `hardware`, sau đó `quant`.

| model | hardware | runtime | workload | quant | prompt | pred | model contrib | hardware contrib | quant contrib |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| `mlx-community-Qwen2.5-32B-Instruct-4bit` | `AMD64 ... (6c) + 15GB` | ollama | concurrent-decode | UD-Q4_K_XL | 117 | **−211,4972** | −88,143 | −155,195 | −63,098 |
| `Qwen3.8-27B-ctx4k` | `AMD64 ... (6c) + 15GB` | llama.cpp | long-context-decay | Unknown | 3514 | **−167,0361** | −69,888 | −155,195 | −4,718 |
| `gemma-4-31B-it-Q4_K_M.gguf` | `AMD64 ... (6c) + 15GB` | llama.cpp | concurrent-decode | MXFP4 | 3152 | **−155,8573** | −97,241 | −155,195 | +9,994 |

### 5.6 Toàn bộ 284 case

Danh sách đầy đủ (payload, predicted TPS, flags, tần suất category, percentile prompt, contribution từng feature) nằm tại:

- `reports/final_deployment_benchmark_negative_cases.csv` — **284 dòng**.

Đây là appendix của báo cáo; không phải valid TPS.

---

## 6. Root Cause Analysis

### 6.1 Linear Regression extrapolation — **có, nhưng không phải nguyên nhân chính của 282 case unseen**

**Bằng chứng:**
- 2/751 known case âm dù actual dương (7,16 và 7,05). Đây là in-sample, chứng minh bản thân Linear Regression không có ràng buộc `decode_tps ≥ 0`.
- Predicted min toàn suite: **−211,4972**; predicted max: **567,3014** — vượt actual range (3,76–466,95).
- Prompt contribution tối đa chỉ +19,58 TPS (tại 23.872 token), không đủ giải thích các giá trị −100 đến −211.

**Kết luận:** Extrapolation là **điều kiện nền** (linear model không bị chặn dưới), nhưng với 282/284 case unseen, nguyên nhân trực tiếp là tổ hợp/category chưa thấy chứ không chỉ là “linear model”.

### 6.2 Unseen category / OneHotEncoder behavior — **có, rõ nhất ở B2**

**Bằng chứng:**
- B2: 46/200 âm; **46/46 có `hardware` unseen**; contribution hardware trung bình của các case âm = **0,000**.
- B1: model unseen → contribution model = **0,000**, nhưng 0/200 âm vì hardware đã thấy có contribution trung bình **+42,102**, bù lại.
- B2 negative contribution trung bình: model **−92,209**, hardware **0,000**, quant **−3,933**, workload **+7,478** → tổng âm.

**Kết luận:** `handle_unknown="ignore"` zero-hoá category chưa thấy. Đây không phải “neutral” trong mô hình có intercept; nó loại bỏ toàn bộ effect của feature. Khi effect bị loại bỏ là dương (hardware tốt), dự đoán giảm mạnh; khi effect bị loại bỏ là âm (model), dự đoán tăng. Vì vậy unseen category có thể gây **bias cả hai hướng**, không an toàn.

### 6.3 Unseen feature combination — **có, rõ nhất ở B3 và D**

**Bằng chứng:**
- B3: 116/500 âm; **116/116 có 5-tuple chưa có trong training**; tất cả category đều seen.
- D: 120/500 âm; **120/120 có 5-tuple chưa có trong training**.
- Contribution trung bình của case âm:
  - B3: model −46,686; hardware −62,321; quant −12,850; workload −3,181.
  - D: model −49,880; hardware −64,936; quant −13,827; workload −1,269.
- Ví dụ worst B3: model −92,778 + hardware −71,004 + quant −63,098 + workload −16,422 + runtime −3,001 + prompt +2,615 + intercept 79,306 = **−164,382**.

**Kết luận:** Linear Regression giả định quan hệ additive; với tổ hợp chưa quan sát, tổng các hệ số có thể rơi ra ngoài miền target. Đây là **extrapolation theo tổ hợp**, không phải lỗi inference.

### 6.4 Distribution shift / coverage — **nguyên nhân cấu trúc chính**

**Bằng chứng:**
- Không gian 5-tuple có thể có: 106 × 18 × 3 × 5 × 13 = **372.060**.
- Số 5-tuple thực tế quan sát: **544** → độ phủ **0,146%**.
- UI `app.py` cho phép chọn độc lập từng feature từ danh sách giá trị đã thấy → người dùng có thể tạo tổ hợp **chưa từng có trong training**.
- 120/120 negative ở nhóm D là tổ hợp chưa thấy; 116/116 ở B3; 46/46 ở B2 (hardware unseen).

**Kết luận:** Đây là **distribution shift do tổ hợp**, không phải shift do giá trị đơn lẻ. Mô hình được huấn luyện trên 544 tổ hợp nhưng UI cho phép 372.060 tổ hợp. Phần lớn input hợp lệ theo schema nằm ngoài phân phối huấn luyện.

### 6.5 Inference pipeline / schema mismatch — **không có bằng chứng**

**Bằng chứng:**
- 0 lỗi inference trên 2.190 test.
- Feature names/order khớp `FEATURES`; `pipe.n_features_in_ = 7`.
- `quant` missing → `Unknown` đúng A1.
- `batch_size = 1` được validate.
- Không có NaN/Inf; `predict()` kiểm tra finite.
- Không có `OneHotEncoder` error; unseen category được xử lý theo `handle_unknown="ignore"`.

**Kết luận:** Không thể quy negative prediction cho schema mismatch hay lỗi pipeline. Đây là hành vi của mô hình đã fit.

### 6.6 `prompt_tokens` — **không phải nguyên nhân chính**

- Hệ số: **+0,0008204 TPS/token**.
- Contribution: 0 tại 0 token; +0,840 tại 1.024; +19,585 tại 23.872.
- Nhiều case âm có `prompt_tokens = 0` (ví dụ B2 −21,77; B3 −164,38 với 3188 token; D min −211,50).
- Max contribution +19,585 không đủ tạo −211,50.

**Kết luận:** `prompt_tokens` chỉ đóng góp nhỏ; negative prediction chủ yếu do categorical contributions.

### 6.7 Collinearity caveat

`OneHotEncoder` giữ đủ mọi level + `LinearRegression` có intercept → thiết kế suy biến. Hệ số không duy nhất; phân rã contribution chỉ là một cách đọc. Các kết luận chính dựa trên:
- predicted values;
- tuple_in_training / unseen flags;
- tổng contribution (bằng prediction − intercept, kiểm tra khớp);
- actual values cho known cases.

Không dựa vào ý nghĩa nhân quả của từng hệ số.

---

## 7. Generalization Limitations

1. **Holdout theo `run_id` là nội suy trong phân phối benchmark**, không phải ngoại suy sang cấu hình mới. A1 LR R² 0,7108 trên test theo `run_id`, nhưng test vẫn chứa model/hardware đã thấy.
2. **Unseen model/hardware:** audit này cho thấy B1 lệch mean +16,5 TPS; B2 23% âm. Tài liệu dự án trước đó (README mục 4) cũng ghi nhận LinearRegression kém hơn Dummy khi test unseen model/hardware; các số đó không được tái lập trong audit này, nên chỉ dùng làm tham chiếu.
3. **High-cardinality:** 106 model, 18 hardware; 544 tổ hợp; median số hàng/tuple = 1, max = 74. Nhiều category chỉ xuất hiện một lần → hệ số không ổn định.
4. **`quant` thiếu:** 273/751 hàng (36,35%) thiếu `quant`; A1 gán `Unknown`. Đây là lựa chọn hợp lý để giữ mẫu, nhưng `Unknown` là category riêng, không đại diện cho một quantization cụ thể.
5. **`batch_size` constant = 1:** không có variation, không thể đánh giá ảnh hưởng.
6. **Mô hình tuyến tính additive:** không có interaction, không có non-linearity; không có ràng buộc target dương.
7. **Không có bằng chứng nhân quả:** correlation/permutation importance chỉ là predictive, không phải causal.
8. **Deployment artifact fit trên toàn bộ dữ liệu:** không dùng để báo cáo holdout metrics; các metrics A1/A2 ở mục 4.4 là từ protocol riêng.

---

## 8. Deployment Readiness Assessment

### 8.1 Theo tiêu chí

| Tiêu chí | Trạng thái | Bằng chứng |
|---|---|---|
| Artifact load + schema | **PASS** | Load OK; 7 features đúng thứ tự; 147 coefficients; không warning. |
| Inference errors | **PASS** | 0/2.190 lỗi; không NaN/Inf. |
| Known configurations | **WARNING** | 2/751 âm (0,27%); in-sample MAE 7,8251, R² 0,9661; actual của 2 case là 7,16 và 7,05. |
| Unseen model | **WARNING** | 0/200 âm nhưng mean 121,353 vs known 104,806 (+16,5 TPS); model contribution bị zero. |
| Unseen hardware | **FAIL** | 46/200 âm (23,0%); hardware contribution = 0. |
| Unseen feature combination | **FAIL** | 116/500 âm (23,2%); tất cả tuple chưa thấy. |
| Randomized valid inputs | **FAIL** | 120/500 âm (24,0%); tất cả tuple chưa thấy. |
| Boundary inputs | **PASS** | 0/39 âm; range 11,432–135,513 TPS. |
| Holdout metrics A1 | **PASS** | Tái lập đúng: MAE 24,6859, RMSE 40,3109, R² 0,7108. |
| UI guard cho negative | **FAIL** | `app.py` hiển thị trực tiếp `pred:.2f token/s`; không có kiểm tra/clip/reject. |

### 8.2 Kết luận sẵn sàng demo

- **Demo có kiểm soát: WARNING — chấp nhận được với điều kiện.**
- **Demo tự do / tuyên bố dự đoán mọi cấu hình: FAIL.**

**Điều kiện bắt buộc nếu demo:**
1. Chỉ dùng cấu hình **đã có trong dataset** và đã kiểm tra trước cho ra dự đoán dương.
2. **Tránh 2 cấu hình known âm** ở mục 5.2; tránh các cấu hình known có pred < 5 TPS (5 hàng) và pred < 10 TPS (8 hàng) nếu muốn demo “đẹp”.
3. Không chọn ngẫu nhiên các selectbox; chuẩn bị payload trước và test bằng `src.inference.predict`.
4. Nếu xuất hiện negative, **không trình bày như TPS**; nói rõ đó là dự đoán không hợp lệ do extrapolation.
5. Giải thích rõ: deployment artifact fit trên toàn bộ dữ liệu; các metrics A1/A2 là holdout riêng; không tuyên bố generalization cho unseen model/hardware.

### 8.3 Danh sách known config rủi ro (pred < 5 TPS)

| Pred | Actual | Model | Hardware | Runtime | Workload | Quant | Prompt |
|---:|---:|---|---|---|---|---|---:|
| −16,7726 | 7,16 | `mlx-community/Qwen3-32B-4bit` | M3 Pro (18-core GPU) + 36GB unified | mlx | chat-short | Unknown | 110 |
| −0,7657 | 7,05 | `mlx-community/Qwen2.5-32B-Instruct-4bit` | M3 Pro (18-core GPU) + 36GB unified | mlx | chat-short | Unknown | 131 |
| 3,4862 | 10,01 | `Flint-40B-A2.5B-UD-cpumoe12-ctx8k` | RTX A4000 (16GB) + AMD EPYC 7453 28-Core Processor (56c) + 504GB | llama.cpp | chat-long | Q3_K_XL | 3014 |
| 3,7888 | 14,41 | `mlx-community-Qwen2.5-14B-Instruct-4bit` | M3 Pro (18-core GPU) + 36GB unified | mlx | chat-short | 4bit | 131 |
| 4,5386 | 10,04 | `Flint-40B-A2.5B-UD-cpumoe12-ctx8k` | RTX A4000 (16GB) + AMD EPYC 7453 28-Core Processor (56c) + 504GB | llama.cpp | agent-trace | Q3_K_XL | 3192 |

---

## 9. Recommendations

### 9.1 Trước buổi thuyết trình (không sửa code)

1. **Chuẩn bị 2–3 payload known-positive** và test trước bằng `src.inference.predict`; lưu kết quả/screenshot.
2. **Không demo chế độ chọn tự do**; nếu demo UI, hướng dẫn người xem chọn đúng cấu hình đã chuẩn bị.
3. Chuẩn bị câu giải thích negative prediction: “Linear Regression không có ràng buộc dương; với tổ hợp chưa quan sát, mô hình có thể extrapolate ra giá trị âm — giá trị này không phải TPS hợp lệ.”
4. Trình bày **holdout A1/A2 riêng** với deployment artifact; không nói “deployment model có R² 0,711” như thể là holdout của artifact.
5. Nêu rõ giới hạn: chỉ nội suy trong phân phối benchmark; không dự đoán cho PC bất kỳ; không kết luận “chạy được/không chạy được”.

### 9.2 Ngắn hạn (cần sửa code — ngoài phạm vi audit này)

1. **Guard negative prediction** trong `app.py`/`src/inference.py`: reject hoặc clip về 0 kèm cảnh báo rõ; không hiển thị negative như TPS.
2. **Cảnh báo unseen combination**: kiểm tra 5-tuple có trong training; nếu không, hiển thị cảnh báo “ngoài phân phối huấn luyện”.
3. **Giới hạn UI** trong demo: chỉ cho chọn các tổ hợp đã quan sát, hoặc thêm chế độ “known configurations only”.
4. **Thêm test tự động** cho: negative prediction, unseen category, unseen combination, schema mismatch.
5. **Bổ sung metadata** cho artifact: version, git commit, sklearn version, ngày fit, hash dataset.

### 9.3 Trung hạn (cải thiện mô hình — ngoài phạm vi audit này)

1. Thử mô hình có ràng buộc/regularization hoặc non-linear (RandomForest/GradientBoosting) — nhưng phải đánh giá lại holdout theo `run_id`.
2. Xử lý `quant` missing tốt hơn (missing indicator, imputation có lý do).
3. Parse `hardware` thành feature cấu trúc (VRAM, GPU type, CPU, RAM) thay vì chuỗi tự do.
4. Thêm interaction hoặc target transform (log) để giảm extrapolation.
5. Thu thập thêm dữ liệu cho các tổ hợp hiếm; đánh giá holdout theo `model`/`hardware` chưa thấy.
6. Không dùng permutation importance như causal effect.

### 9.4 Tài liệu

- Cập nhật README/limitations: negative prediction behavior, coverage 0,146%, unseen combination risk.
- Ghi rõ deployment artifact ≠ holdout metrics.
- Đính kèm báo cáo này và CSV negative cases khi nộp.

---

## Appendix A — Evidence files

| File | Nội dung |
|---|---|
| `reports/final_deployment_benchmark.md` | Báo cáo này |
| `reports/final_deployment_benchmark_group_stats.csv` | Thống kê 6 nhóm test |
| `reports/final_deployment_benchmark_negative_cases.csv` | **284** negative case, full payload + flags + contribution |
| `models/cdk_a1_deployment.joblib` | Artifact được audit (không sửa) |
| `data/raw/llm-speed-runs-2026-10-22.6.csv` | Dataset gốc (không sửa) |

**Hash:**

| File | SHA-256 |
|---|---|
| `models/cdk_a1_deployment.joblib` | `b112a9bada0383e134c32a055f82e0c2b40e1a26c50d0a5bd55aa8bbd70d192d` |
| `data/raw/llm-speed-runs-2026-10-22.6.csv` | `0b7a57d6f2a8ef768f161df143d665c2a3723dfb21b426a289ea4e062f522eb2` |

## Appendix B — Repro note

- Benchmark dùng `src.inference.predict` và artifact hiện tại.
- Holdout dùng đúng protocol `GroupShuffleSplit` theo `run_id`, `test_size=0.2`, `random_state=42`.
- Không retrain/chỉnh sửa artifact deployment; không sửa code/dataset/notebook.
- Các con số trong báo cáo lấy từ lần chạy audit ngày 2026-11-07; nếu artifact/dataset thay đổi, phải chạy lại.
