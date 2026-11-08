# CDK — Đánh giá phương án Log Transformation để loại bỏ dự đoán TPS âm

**Ngày:** 2026-11-07  
**Loại nhiệm vụ:** Research & evaluation. **Không triển khai thay đổi production.**  
**Trạng thái:** Không sửa `app.py`, `src/inference.py`, `models/cdk_a1_deployment.joblib`, `data/raw/`, feature set A1, split protocol hay baseline A1. Không `git add`, commit, push.  
**Dataset:** `data/raw/llm-speed-runs-2026-10-22.6.csv` — 751 rows, 106 model, 18 hardware, 335 `run_id`; target `decode_tps` min 3.76, max 466.95, 0 giá trị ≤ 0.  
**SHA-256 dataset:** `0b7a57d6f2a8ef768f161df143d665c2a3723dfb21b426a289ea4e062f522eb2`  
**SHA-256 artifact tham chiếu:** `b112a9bada0383e134c32a055f82e0c2b40e1a26c50d0a5bd55aa8bbd70d192d` (không load/ghi đè)  
**SHA-256 `src/inference.py`:** `cf8c4d180c9907263762d19fca964405758f2736862ef0b5f3dde0da5a568a60`  
**Môi trường:** Python 3.11.9, pandas 3.0.3, numpy 2.4.6, scikit-learn 1.9.0, scipy 1.17.1.

> Báo cáo chỉ trả lời câu hỏi nghiên cứu. Không có thay đổi nào được áp dụng vào CDK production.

---

## 0. Tóm tắt kết luận

| Câu hỏi | Kết luận ngắn | Bằng chứng chính |
|---|---|---|
| B có loại bỏ hoàn toàn dự đoán âm? | **Có, trên mọi protocol và mọi seed đã chạy.** | 0 âm / 8918 dự đoán test (20 seeds × 3 protocol); A có 199 âm. |
| B có cải thiện MAE? | **Có, trung bình.** | `run_id`: 26.0335 so với 27.7618; 18/20 seed tốt hơn. |
| B có cải thiện RMSE/R² trên protocol chính? | **Không, tệ hơn.** | `run_id`: RMSE 54.5874 so với 51.9526; R² 0.5423 so với 0.5871. |
| B có dự đoán TPS cao tốt hơn? | **Không, tệ hơn rõ rệt.** | Seed 42, actual ≥ 200: `run_id` MAE 56.24 so với 48.59; `model` 112.98 so với 81.98. |
| B có khắc phục underprediction? | **Không, làm nặng hơn.** | Mean bias `run_id` −11.31 so với −5.62 TPS; `hardware` −31.81 so với −16.11. |
| B có tổng quát hóa tốt hơn sang `hardware` chưa thấy? | **Không đủ để kết luận; cả hai đều rất kém.** | R² `hardware`: B −3.00 so với A −6.56 (đều âm mạnh, test set 16–348 rows). |
| B có phù hợp production? | **Không.** | Đánh đổi RMSE/R² và bias hệ thống để đổi lấy positivity; positivity không đồng nghĩa correctness. |
| Có nên coi B là candidate tiếp theo? | **Không.** | Không dominate A1 trên protocol chính; sai số TPS cao và bias trung bình lớn hơn. |

**Kết luận bắt buộc (chọn một): Chưa đủ cơ sở, cần nghiên cứu phương pháp khác.**

A1 tiếp tục là model production hiện tại (không thay đổi), nhưng vấn đề dự đoán âm **chưa được giải quyết một cách đáng tin cậy**. Log Transformation là một hướng hợp lý để nghiên cứu thêm, nhưng kết quả hiện tại chưa đủ để đưa vào production hoặc coi là candidate kế tiếp.

---

## 1. Bối cảnh và câu hỏi nghiên cứu

A1 là Linear Regression dự đoán trực tiếp `decode_tps` từ 7 feature:

`model`, `hardware`, `runtime`, `workload`, `quant`, `prompt_tokens`, `batch_size`.

Vì Linear Regression không có ràng buộc đầu ra dương, A1 có thể dự đoán TPS âm — đặc biệt khi gặp tổ hợp feature chưa thấy hoặc `hardware` chưa thấy. Báo cáo `reports/final_deployment_benchmark.md` đã ghi nhận 284 dự đoán âm trên bộ kiểm thử deployment, tập trung ở các cấu hình chưa quan sát.

Câu hỏi nghiên cứu:

1. Huấn luyện Linear Regression trên `log(decode_tps)` rồi chuyển ngược bằng `exp()` (phương án B) có loại bỏ được dự đoán âm không?
2. B có duy trì hoặc cải thiện chất lượng dự đoán trên thang TPS gốc không?
3. B có tổng quát hóa tốt hơn A1 sang `model`/`hardware` chưa thấy không?
4. Nếu B không tốt hơn, có hướng tiếp cận nào khác đáng nghiên cứu không?

---

## 2. Phương pháp thực nghiệm

### 2.1. Dữ liệu và tiền xử lý

- Giữ nguyên A1: target `decode_tps`; categorical `model`, `hardware`, `runtime`, `workload`, `quant`; numeric `prompt_tokens`, `batch_size`.
- `quant` thiếu → `Unknown`.
- `OneHotEncoder(handle_unknown="ignore")` cho categorical; numeric passthrough.
- Không dùng `model_size`, không thêm feature mới, không đổi hyperparameter.
- Không sửa dữ liệu gốc; mọi thao tác trên bản sao trong bộ nhớ.

### 2.2. Hai phương án

| | Phương án A (baseline A1) | Phương án B (log target) |
|---|---|---|
| Model | `LinearRegression` | `LinearRegression` |
| Target huấn luyện | `decode_tps` | `log(decode_tps)` |
| Dự đoán | trực tiếp TPS | `exp(linear predictor)` |
| Metrics | tính trên TPS gốc | tính trên TPS gốc |
| Preprocessing | giống hệt A1 | giống hệt A1 |

B có thêm một biến thể chẩn đoán **B + Duan smearing** (`pred_B × mean(exp(residual_log))`) để kiểm tra retransformation bias. Đây chỉ là chẩn đoán, không phải phương án production.

### 2.3. Protocol chia tập

Giữ nguyên thiết kế A1 đã công bố:

- `GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)`.
- `groups = run_id` cho protocol chính.
- `groups = model` cho kiểm tra model chưa thấy.
- `groups = hardware` cho kiểm tra hardware chưa thấy.
- A và B dùng **cùng một train/test split** ở mỗi seed.
- Seeds: **42–61** (20 seeds).

Kiểm tra: giao giữa group train và test = 0 ở mọi seed/protocol.

### 2.4. Metrics và kiểm tra tính hợp lệ

Trên TPS gốc:

- MAE, RMSE, R².
- Số dự đoán âm, bằng 0, không hữu hạn (NaN/Inf).
- Số dự đoán < 1 TPS, < min của train, > max của train, > 466.95 (max toàn dataset), > 500.
- Phân phối: min, p01, p05, median, p95, p99, max, mean, std.
- Mean bias (`mean_pred − mean_actual`), median bias, mean ratio (`mean_pred / mean_actual`).

Không thay thế dự đoán âm bằng 0 hoặc bất kỳ giá trị TPS tùy ý nào. Mọi số liệu âm được báo cáo nguyên trạng.

### 2.5. Kiểm định thống kê

- Paired t-test và Wilcoxon signed-rank trên chênh lệch MAE (B − A) theo từng seed.
- Bootstrap 95% CI cho mean chênh lệch (10.000 lần, seed 42).

**Lưu ý quan trọng:** 20 seed là 20 **cách chia lại cùng một tập 751 rows**, không phải 20 mẫu độc lập từ một quần thể. Vì vậy p-value chỉ mô tả độ nhạy của kết quả với cách chia tập, **không** chứng minh ý nghĩa thống kê ở cấp quần thể. Báo cáo không tuyên bố "có ý nghĩa thống kê" cho generalization.

### 2.6. Xử lý tràn số

- `exp()` được bọc trong `np.errstate(over="ignore", under="ignore", invalid="ignore")`.
- Metrics tự động phát hiện và đếm non-finite; nếu có non-finite, metrics được đánh dấu `None` thay vì tính sai.
- Trong toàn bộ thực nghiệm, **không có** NaN/Inf nào xuất hiện ở A hoặc B.

---

## 3. Thông tin tái lập kết quả

A1 được tái tạo đúng đến sai số số học máy (max abs diff < 1e-14) so với kết quả đã công bố trong `agent_workspace/reports/model_generalization_results.csv`.

| Protocol | MAE kỳ vọng | MAE tái lập | RMSE kỳ vọng | RMSE tái lập | R² kỳ vọng | R² tái lập | Khớp |
|---|---:|---:|---:|---:|---:|---:|---|
| `run_id` (seed 42) | 24.6858606933 | 24.6858606933 | 40.3109074165 | 40.3109074165 | 0.7107603562 | 0.7107603562 | ✅ |
| `model` (seed 42) | 59.4504771400 | 59.4504771400 | 68.5815412893 | 68.5815412893 | −0.2430974944 | −0.2430974944 | ✅ |
| `hardware` (seed 42) | 72.1330424919 | 72.1330424919 | 89.2587354326 | 89.2587354326 | −0.1940441958 | −0.1940441958 | ✅ |

Thông tin tái lập:

- Script thực nghiệm: `C:\Users\Admin\AppData\Local\Temp\opencode\cdk_log\evaluate_log.py` (temp, không nằm trong repo).
- Kết quả thô: `ab_results.csv`, `ab_seed42_predictions.csv`, `ab_bins_seed42.csv`, `ab_paired_tests.csv`, `ab_summary.json`.
- Chẩn đoán 284 negative cases: `negative_case_diagnostic.py`, `negative_cases_ab_predictions.csv`, `negative_cases_ab_summary.json`.
- Không có artifact production nào bị ghi đè; hash dataset/artifact/`src/inference.py` không đổi.

---

## 4. Bảng so sánh A/B — seed 42 (protocol chính `run_id`)

| Variant | n_test | MAE | RMSE | R² | Âm | < 1 TPS | pred min | pred max | mean bias | mean ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A_direct | 156 | 24.6859 | 40.3109 | 0.7108 | 0 | 0 | 9.7858 | 329.8405 | +3.7416 | 1.0352 |
| B_log | 156 | **22.2586** | 43.2535 | 0.6670 | 0 | 0 | 16.3480 | 315.2642 | −6.5195 | 0.9388 |
| B_log_smear | 156 | 22.4050 | 43.1842 | 0.6681 | 0 | 0 | 16.6209 | 320.5262 | −4.8516 | 0.9544 |

Nhận xét seed 42:

- B giảm MAE **2.43 TPS** (−9.8%) nhưng tăng RMSE **2.94 TPS** (+7.3%) và giảm R² **0.0438**.
- B chuyển từ overprediction nhẹ (+3.74 TPS) sang underprediction (−6.52 TPS).
- Smearing chỉ kéo mean bias về −4.85 TPS; không sửa được hướng sai lệch.

---

## 5. Kết quả 20 seeds (42–61)

### 5.1. Trung bình ± độ lệch chuẩn theo seed

| Protocol | Variant | MAE | RMSE | R² | Tổng âm | Tổng < 1 TPS | Mean bias | Mean ratio |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `run_id` | A_direct | 27.7618 ± 5.6332 | **51.9526 ± 16.1425** | **0.5871 ± 0.1751** | 10 | 10 | −5.62 | 0.9484 |
| `run_id` | B_log | **26.0335 ± 6.4794** | 54.5874 ± 17.5677 | 0.5423 ± 0.2023 | 0 | 0 | −11.31 | 0.8955 |
| `run_id` | B_log_smear | 26.1766 ± 6.4957 | 54.4900 ± 17.4428 | 0.5447 ± 0.1993 | 0 | 0 | −9.77 | 0.9101 |
| `model` | A_direct | 64.7528 ± 12.1392 | **81.1975 ± 15.2993** | 0.0241 ± 0.2628 | 8 | 8 | +6.64 | 1.0917 |
| `model` | B_log | **60.1882 ± 13.4295** | 81.2211 ± 18.0967 | **0.0474 ± 0.1977** | 0 | 0 | −16.57 | 0.8708 |
| `model` | B_log_smear | 60.3816 ± 13.3401 | 80.9862 ± 17.9213 | 0.0519 ± 0.1991 | 0 | 0 | −15.04 | 0.8854 |
| `hardware` | A_direct | 72.1369 ± 29.5730 | 99.1630 ± 46.1508 | −6.5556 ± 28.0268 | 181 | 199 | −16.11 | 1.1539 |
| `hardware` | B_log | **64.4587 ± 29.5892** | **95.6118 ± 48.9953** | **−3.0002 ± 12.5950** | 0 | 0 | −31.81 | 0.8929 |
| `hardware` | B_log_smear | 64.3316 ± 29.7203 | 95.4260 ± 48.9669 | −3.1062 ± 13.0997 | 0 | 0 | −30.79 | 0.9076 |

### 5.2. Median theo seed (ít bị ảnh hưởng bởi outlier)

| Protocol | Variant | MAE median | RMSE median | R² median |
|---|---|---:|---:|---:|
| `run_id` | A_direct | 27.3861 | **47.1202** | **0.6077** |
| `run_id` | B_log | **24.6355** | 50.5664 | 0.5816 |
| `model` | A_direct | 65.1668 | 80.8551 | **0.0790** |
| `model` | B_log | **56.9105** | **77.8566** | 0.0531 |
| `hardware` | A_direct | 64.9061 | 88.2918 | **−0.2006** |
| `hardware` | B_log | **60.0990** | **86.6083** | −0.2572 |

Kết luận từ median nhất quán với mean: B tốt hơn MAE, nhưng không tốt hơn RMSE/R² trên `run_id`; trên `hardware`, R² median của B còn âm hơn A.

### 5.3. Chênh lệch paired B − A (MAE) và kiểm định

| Protocol | Mean ΔMAE | SD | Min | Max | B tốt hơn | Bootstrap 95% CI | Paired t p | Wilcoxon p |
|---|---:|---:|---:|---:|---:|---|---:|---:|
| `run_id` | −1.7283 | 2.3903 | −6.2207 | +3.6903 | 18/20 | [−2.7123, −0.6499] | 0.0044 | 0.0032 |
| `model` | −4.5646 | 7.4382 | −15.5006 | +9.7194 | 13/20 | [−7.6925, −1.4013] | 0.0129 | 0.0240 |
| `hardware` | −7.6781 | 11.5642 | −39.4298 | +9.3448 | 16/20 | [−12.8626, −3.0516] | 0.0079 | 0.0049 |

Diễn giải đúng:

- Trên 20 cách chia lại cùng dataset, B **có xu hướng** giảm MAE.
- Tuy nhiên đây không phải bằng chứng quần thể: các seed không độc lập, test set chồng lấp đáng kể, và kết quả RMSE/R² đi ngược lại.
- B + smearing chỉ cải thiện nhẹ MAE so với B (ví dụ `run_id` −1.5852 so với −1.7283), cho thấy smearing không giải quyết được nguyên nhân bias.

### 5.4. Độ ổn định theo seed

- `run_id`: B tốt hơn MAE ở 18/20 seed, nhưng RMSE của B chỉ tốt hơn ở **3/20 seed** (xấu hơn 17/20) và R² cũng chỉ tốt hơn ở **3/20 seed** (xấu hơn 17/20).
- `model`: B tốt hơn MAE ở 13/20 seed; RMSE gần như tương đương trung bình (81.22 so với 81.20).
- `hardware`: test size dao động **16–348 rows**; R² rất bất ổn định. Seed 54 có A R² = **−125.6**, cho thấy R² trên protocol này không phải thước đo đáng tin cậy.

---

## 6. Kết quả generalization

Protocol giữ nguyên thiết kế A1 trước đó: `GroupShuffleSplit` theo `model` và theo `hardware`, `test_size=0.2`, seed 42 (và mở rộng 20 seeds).

### 6.1. Seed 42 — model chưa thấy

| Variant | n_test | MAE | RMSE | R² | Âm | Mean bias | Mean ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| A_direct | 147 | 59.4505 | 68.5815 | −0.2431 | 0 | +36.50 | 1.4180 |
| B_log | 147 | **47.4363** | **57.3375** | **0.1311** | 0 | +11.55 | 1.1322 |
| B_log_smear | 147 | 48.1064 | 57.6962 | 0.1202 | 0 | +12.93 | 1.1481 |

B cải thiện rõ ở seed 42, nhưng qua 20 seeds lợi thế MAE co lại còn −4.56 và RMSE gần như bằng A. R² vẫn rất thấp (0.047 trung bình), tức mô hình **không thực sự tổng quát hóa** sang model mới; B chỉ bớt tệ hơn A.

### 6.2. Seed 42 — hardware chưa thấy

| Variant | n_test | MAE | RMSE | R² | Âm | Mean bias | Mean ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| A_direct | 19 | 72.1330 | 89.2587 | −0.1940 | 0 | +29.47 | 1.5745 |
| B_log | 19 | **59.6232** | **85.7367** | **−0.1017** | 0 | +11.00 | 1.2145 |
| B_log_smear | 19 | 60.3122 | 86.0139 | −0.1088 | 0 | +12.03 | 1.2346 |

Qua 20 seeds, test size 16–348 rows, R² trung bình A −6.56 và B −3.00. Cả hai đều **không đủ năng lực dự đoán** hardware chưa thấy. B giảm mức độ tệ nhưng không tạo ra khả năng tổng quát hóa.

### 6.3. Diễn giải

- B không tạo ra thông tin mới về `model`/`hardware`; nó chỉ thay đổi hàm mất mát và ép đầu ra dương.
- Với `OneHotEncoder(handle_unknown="ignore")`, category chưa thấy bị zero hóa. Cả A và B đều dựa vào intercept và các feature còn lại, nên không thể suy ra TPS cho cấu hình hoàn toàn mới.
- Positivity của B là **do cấu trúc toán học** (`exp` luôn dương), không phải do mô hình hiểu đúng vật lý/hiệu năng của hardware mới.

---

## 7. Phân tích tính hợp lệ của dự đoán

### 7.1. Âm / 0 / không hữu hạn (tổng 20 seeds × 3 protocol)

| Protocol | Variant | Tổng dự đoán test | Âm | Bằng 0 | Non-finite | < 1 TPS | < train min | > train max | > 466.95 | > 500 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `run_id` | A_direct | 3064 | 10 | 0 | 0 | 10 | 10 | 0 | 0 | 0 |
| `run_id` | B_log | 3064 | **0** | 0 | 0 | **0** | 2 | 2 | 0 | 0 |
| `model` | A_direct | 3056 | 8 | 0 | 0 | 8 | 8 | 0 | 0 | 0 |
| `model` | B_log | 3056 | **0** | 0 | 0 | **0** | 0 | 0 | 0 | 0 |
| `hardware` | A_direct | 2798 | 181 | 0 | 0 | 199 | 250 | 0 | 0 | 0 |
| `hardware` | B_log | 2798 | **0** | 0 | 0 | **0** | 0 | 0 | 0 | 0 |

Tổng: A có **199 dự đoán âm** trên 8918 dự đoán test; B có **0**. B cũng không tạo ra NaN/Inf, không có giá trị bằng 0, không có dự đoán < 1 TPS.

Tuy nhiên, B có **2 dự đoán vượt max của train** trên `run_id` (và không vượt max toàn dataset 466.95). Đây là dấu hiệu extrapolation nhẹ, cần guard nếu triển khai.

### 7.2. Phân phối dự đoán (tổng hợp 20 seeds)

| Protocol | Variant | min | p01 | median | p99 | max | mean |
|---|---|---:|---:|---:|---:|---:|---:|
| `run_id` | A_direct | −37.917 | −12.428 | 87.670 | 330.993 | 334.671 | 101.138 |
| `run_id` | B_log | 8.078 | 10.729 | 71.567 | 367.721 | 374.671 | 95.449 |
| `model` | A_direct | −8.843 | −8.573 | 115.773 | 235.437 | 235.437 | 115.062 |
| `model` | B_log | 13.917 | 13.917 | 89.213 | 225.855 | 225.863 | 91.851 |
| `hardware` | A_direct | −47.931 | −45.681 | 78.880 | 244.114 | 244.196 | 77.651 |
| `hardware` | B_log | 11.272 | 19.485 | 57.443 | 171.916 | 185.470 | 61.947 |

Nhận xét:

- B **nén phân phối về phía thấp**: median và mean đều thấp hơn A ở cả ba protocol.
- B không tạo ra outlier cao bất thường trong thực nghiệm này, nhưng phân phối bị lệch xuống rõ rệt.
- A có đuôi âm; B có đuôi dương nhưng **không đảm bảo đúng độ lớn**.

### 7.3. Sai số theo dải TPS thực (seed 42)

| Protocol | Dải actual | n | A MAE | B MAE | A mean bias | B mean bias |
|---|---|---:|---:|---:|---:|---:|
| `run_id` | [10,30) | 1 | 5.82 | **0.74** | −5.82 | +0.74 |
| `run_id` | [30,60) | 47 | 22.41 | **13.87** | +17.81 | +12.64 |
| `run_id` | [60,100) | 60 | 13.68 | **7.22** | +10.87 | +4.80 |
| `run_id` | [100,200) | 16 | **26.01** | 36.69 | +0.55 | −21.29 |
| `run_id` | [200,500] | 32 | **48.59** | 56.24 | −28.40 | −48.72 |
| `model` | [0,10) | 1 | 34.19 | **10.55** | +34.19 | +10.55 |
| `model` | [10,30) | 13 | 25.39 | **12.74** | +25.39 | +10.96 |
| `model` | [30,60) | 37 | 50.90 | **32.23** | +50.90 | +32.23 |
| `model` | [60,100) | 61 | 77.58 | **48.28** | +77.58 | +48.28 |
| `model` | [100,200) | 21 | **29.11** | 51.33 | −22.21 | −48.16 |
| `model` | [200,500] | 14 | **81.98** | 112.98 | −81.98 | −112.98 |
| `hardware` | [0,10) | 2 | 57.30 | **47.19** | +57.30 | +47.19 |
| `hardware` | [10,30) | 13 | 56.67 | **37.21** | +56.67 | +37.21 |
| `hardware` | [30,60) | 1 | 102.07 | **92.80** | +102.07 | +92.80 |
| `hardware` | [60,100) | 1 | **11.82** | 24.80 | +11.82 | −24.80 |
| `hardware` | [200,500] | 2 | **202.66** | 218.55 | −202.66 | −218.55 |

Đây là bằng chứng cốt lõi: **B tốt hơn ở dải thấp/trung bình nhưng tệ hơn ở dải cao**. Vì RMSE bị chi phối bởi các điểm TPS cao, B làm RMSE xấu đi dù MAE tốt hơn.

### 7.4. Chẩn đoán trên 284 payload từng cho dự đoán âm

Fit A và B trên **toàn bộ 751 rows**, dự đoán lại đúng 284 payload trong `reports/final_deployment_benchmark_negative_cases.csv`.

- A refit khớp `predicted_tps` trong CSV với max abs diff **1.42e-14** → xác nhận đúng pipeline.
- B cho **0 âm / 0 non-finite** trên 284 payload.
- Phân phối B: min 2.04, median 18.53, max 81.77 TPS.
- Theo nhóm: known 6.71 (median), unseen hardware 29.12, unseen feature combination 16.07, random valid 15.93.

Hai payload **có actual đã biết**:

| Model | Actual TPS | A dự đoán | B dự đoán | B + smearing |
|---|---:|---:|---:|---:|
| `mlx-community/Qwen3-32B-4bit` | 7.16 | **−16.77** | **6.91** | 7.02 |
| `mlx-community/Qwen2.5-32B-Instruct-4bit` | 7.05 | **−0.77** | **6.51** | 6.62 |

Đây là bằng chứng định lượng rằng B **sửa đúng** các trường hợp TPS thấp đã biết, không chỉ "làm cho dương". Tuy nhiên, 282 payload còn lại **không có actual**, nên không thể kết luận B dự đoán đúng cho unseen hardware/tổ hợp chưa thấy. Positivity không đồng nghĩa correctness.

---

## 8. Phân tích

### 8.1. Log Transformation có thực sự cải thiện chất lượng dự đoán?

**Kết luận ngắn: Có cải thiện một phần, nhưng không toàn diện.**

- B giảm MAE trung bình trên cả ba protocol và loại bỏ hoàn toàn dự đoán âm.
- B **không** cải thiện RMSE/R² trên protocol chính `run_id`; RMSE xấu hơn ở 17/20 seed.
- B **làm nặng thêm** underprediction trung bình, đặc biệt ở TPS cao.
- Vì vậy B không thể được mô tả là "tốt hơn A1"; chính xác hơn là **đánh đổi (trade-off)**: đổi positivity và MAE lấy RMSE/R² và bias.

### 8.2. Vì sao tối ưu sai số trong không gian log khác tối ưu MAE/RMSE trên TPS?

Linear Regression trên `log(y)` tối thiểu hóa:

\[
\sum_i (\log y_i - \hat z_i)^2
\]

Đây là sai số **tương đối/bội** (multiplicative), không phải sai số tuyệt đối trên TPS. Hệ quả:

- Một sai lệch 10% ở TPS 10 (1 TPS) và 10% ở TPS 300 (30 TPS) được đối xử gần như nhau trong log space.
- Vì vậy mô hình dồn nỗ lực fit các giá trị nhỏ, nơi sai số tương đối dễ thấy; các giá trị lớn bị "co" lại.
- Trên thang TPS, điều này làm giảm MAE (nhiều điểm thấp/trung bình) nhưng tăng RMSE (các điểm cao bị dự đoán thiếu mạnh).
- MAE và RMSE là hai hàm mất mát khác nhau: MAE tối ưu median có điều kiện; RMSE tối ưu mean có điều kiện. Log-space OLS không trực tiếp tối ưu cả hai.

### 8.3. `exp()` chuyển ngược ảnh hưởng đến ước lượng TPS trung bình thế nào?

Nếu `Z = log(Y)` và mô hình dự đoán `Ẑ = E[Z | X]`, thì:

\[
\exp(\hat Z) = \exp(E[\log Y | X]) \approx \text{median}(Y | X)
\]

trong khi TPS trung bình cần:

\[
E[Y | X] = \exp\left(E[\log Y|X] + \frac{1}{2}\sigma^2\right)
\]

Theo bất đẳng thức Jensen, `E[exp(Z)] ≥ exp(E[Z])`. Vì vậy `exp(Ẑ)` **thiên về thấp hơn** TPS trung bình thật. Trong thực nghiệm:

- `run_id`: mean ratio B = 0.8955 (dự đoán chỉ bằng ~89.6% actual trung bình).
- `model`: 0.8708.
- `hardware`: 0.8929.
- A có mean ratio gần 1 hơn (0.9484, 1.0917, 1.1539).

Đây là underprediction hệ thống, không phải nhiễu ngẫu nhiên.

### 8.4. Retransformation bias là gì và ảnh hưởng ra sao?

Retransformation bias là sai lệch phát sinh khi ta dự đoán trên thang biến đổi (log) rồi chuyển ngược về thang gốc bằng hàm phi tuyến (`exp`), nhưng không hiệu chỉnh đúng kỳ vọng.

- Nếu residuals trong log space có phương sai σ² và phân phối xấp xỉ normal, hệ số hiệu chỉnh lý thuyết là `exp(σ²/2)`.
- Duan smearing ước lượng hệ số này bằng `mean(exp(residual))` trên train.
- Trong thực nghiệm, smear factor chỉ khoảng **1.016** (1.6%). Nó kéo mean bias từ −11.31 lên −9.77 TPS (`run_id`), tức chỉ sửa được ~1.5 TPS.
- Phần bias còn lại đến từ **model misspecification** và **phân phối không đồng nhất**, không phải chỉ từ lognormal variance. Vì vậy smearing không đủ.

Lưu ý: smearing giả định residual độc lập với feature và phân phối ổn định; các giả định này không được kiểm chứng đầy đủ ở đây. B + smearing chỉ là chẩn đoán, không phải phương án production.

### 8.5. Phương án B có phù hợp hơn A1 cho production không?

**Không.**

Lý do:

1. **Không dominate trên protocol chính:** RMSE và R² của B xấu hơn A trên `run_id`, protocol đại diện cho phân phối benchmark.
2. **Sai số TPS cao tệ hơn:** Ở dải 200–500 TPS, B có MAE cao hơn A ở cả ba protocol (seed 42).
3. **Bias hệ thống lớn hơn:** B underprediction trung bình 10–32 TPS tùy protocol, nặng hơn A.
4. **Generalization không được cải thiện thực chất:** R² vẫn âm trên `hardware`; B chỉ giảm mức độ tệ.
5. **Positivity là tính chất toán học, không phải bằng chứng đúng:** `exp()` luôn dương, nhưng có thể dự đoán sai rất xa. Không được đánh đổi accuracy để có positivity.
6. **Rủi ro vận hành mới:** `exp()` có thể tràn số với input cực đoan; cần guard bổ sung. Trong thực nghiệm chưa xảy ra, nhưng production phải xử lý.

### 8.6. Nếu B không tốt hơn, hướng nào đáng nghiên cứu?

Các hướng có cơ sở lý thuyết và nên được thử trong nghiên cứu tiếp theo (chưa triển khai):

1. **GLM với log link và phân phối Gamma/Tweedie/Poisson** — mô hình hóa trực tiếp mean dương trên thang gốc, tránh retransformation bias; phù hợp với target TPS dương, lệch phải.
2. **Quantile Regression** (ví dụ median hoặc quantile thấp) — cho dự đoán có kiểm soát phân vị, không ép qua `exp`.
3. **Hai giai đoạn / hurdle model** — phân loại "cấu hình có hợp lệ để dự đoán" trước, chỉ regress khi đủ điều kiện; kết hợp OOD detection.
4. **Robust regression / weighted least squares** — giảm ảnh hưởng của extrapolation và outlier, có thể giảm dự đoán âm mà không đổi target.
5. **Mô hình cây (Gradient Boosting/Random Forest) với ràng buộc dương** — học quan hệ phi tuyến, giảm extrapolation tuyến tính; nhưng cần kiểm soát overfit và unseen category.
6. **Applicability domain / OOD detection** — từ chối dự đoán khi input nằm ngoài vùng hỗ trợ, thay vì cố trả về một con số dương. Đây là hướng an toàn nhất cho UI.
7. **Bổ sung dữ liệu cho `hardware`/`model` chưa thấy** — giải quyết nguyên nhân gốc thay vì hậu xử lý.

Trong ngắn hạn, guard hiện có trong `app.py` (chặn 5-tuple không hỗ trợ, chặn `prompt_tokens` ngoài range, từ chối `pred <= 0`, cảnh báo mạnh `0 < pred < 5`) là hành vi trung thực và nên được giữ nguyên. **Không clip âm về 0.**

---

## 9. Hạn chế của từng phương án

### 9.1. Phương án A (A1 hiện tại)

- Không có ràng buộc dương; tạo 199 dự đoán âm trên 8918 dự đoán test, tập trung ở `hardware` chưa thấy (181).
- Extrapolation tuyến tính có thể tạo giá trị rất âm (min −47.93 TPS trong thực nghiệm).
- Không có cơ chế từ chối dự đoán ngoài miền hỗ trợ.
- R² âm trên `model`/`hardware` chưa thấy.

### 9.2. Phương án B (log target)

- Không tối ưu trực tiếp MAE/RMSE trên TPS; làm xấu RMSE/R² trên `run_id`.
- Underprediction hệ thống do retransformation bias; smearing chỉ sửa được ~1.6%.
- Sai số lớn hơn A ở dải TPS cao (200–500).
- Generalization sang `hardware` vẫn rất kém (R² trung bình −3.00).
- `exp()` có thể tràn số với input cực đoan; cần guard.
- Positivity không đảm bảo correctness; có thể tạo dự đoán dương nhưng sai xa.
- Không giải quyết được nguyên nhân gốc: thiếu thông tin cho category chưa thấy.

### 9.3. Hạn chế chung của thực nghiệm

- 20 seed là các cách chia lại cùng một dataset, không phải mẫu độc lập; p-value chỉ phản ánh độ nhạy với split.
- `hardware` protocol có test size rất nhỏ và biến động (16–348 rows), R² không ổn định.
- Không có ground truth cho 282/284 payload chẩn đoán; không thể đánh giá accuracy trên unseen hardware.
- Chưa kiểm định giả định phân phối của residual log; chưa thử cross-validation lồng ghép.
- Chưa thử các biến thể regularization, GLM, quantile regression trong báo cáo này.

---

## 10. Kết luận và khuyến nghị

### 10.1. Kết luận (chọn một)

> **Chưa đủ cơ sở, cần nghiên cứu phương pháp khác.**

Giải thích:

- B loại bỏ hoàn toàn dự đoán âm và cải thiện MAE trung bình, nhưng **không vượt A1 một cách toàn diện**: RMSE/R² xấu hơn trên protocol chính, underprediction hệ thống nặng hơn, sai số TPS cao lớn hơn, và generalization sang hardware chưa thấy vẫn rất kém.
- Vì vậy B **không đủ cơ sở để trở thành candidate production kế tiếp**, và cũng **không nên được chấp nhận chỉ vì mọi dự đoán đều dương**.
- A1 tiếp tục là model production hiện tại, nhưng vấn đề dự đoán âm **chưa được giải quyết**; cần nghiên cứu các phương pháp khác (GLM log-link/Gamma/Tweedie, quantile regression, hurdle model, OOD detection, bổ sung dữ liệu).

### 10.2. Khuyến nghị hành động

1. **Không thay đổi production:** giữ A1, giữ artifact `models/cdk_a1_deployment.joblib`, giữ `src/inference.py` và `app.py` như hiện tại.
2. **Giữ guard hiện có:** từ chối `pred <= 0`, cảnh báo mạnh `0 < pred < 5`, chặn input ngoài miền hỗ trợ. Không clip âm về 0.
3. **Không triển khai B** dưới bất kỳ hình thức nào (kể cả sau smearing) trong production.
4. **Nghiên cứu tiếp theo ưu tiên:**
   - GLM log-link với Gamma/Tweedie — so sánh trên cùng protocol `run_id`, `model`, `hardware`.
   - Quantile regression / hurdle model.
   - OOD/applicability-domain detection để từ chối dự đoán thay vì trả số.
   - Bổ sung dữ liệu cho các nhóm `hardware`/`model` còn thiếu.
5. **Tiêu chí chấp nhận candidate mới:** phải không tệ hơn A1 về RMSE và R² trên `run_id`, không làm nặng thêm bias trung bình, đồng thời giảm đáng kể dự đoán âm — không chỉ đạt positivity.
6. **Nếu cần demo:** chỉ demo cấu hình đã có trong dataset, không chọn ngẫu nhiên, giải thích rõ negative prediction là không hợp lệ.

---

## 11. Phụ lục — Tệp và lệnh tái lập

### 11.1. Tệp kết quả (temp, không commit)

```
C:\Users\Admin\AppData\Local\Temp\opencode\cdk_log\
├── evaluate_log.py
├── ab_results.csv
├── ab_seed42_predictions.csv
├── ab_bins_seed42.csv
├── ab_paired_tests.csv
├── ab_summary.json
├── negative_case_diagnostic.py
├── negative_cases_ab_predictions.csv
└── negative_cases_ab_summary.json
```

### 11.2. Lệnh chạy

```powershell
cd "C:\Users\Admin\Desktop\CSI\SPCK"
python "C:\Users\Admin\AppData\Local\Temp\opencode\cdk_log\evaluate_log.py"
python "C:\Users\Admin\AppData\Local\Temp\opencode\cdk_log\negative_case_diagnostic.py"
```

### 11.3. Kiểm tra phạm vi

- `git status --porcelain`: chỉ `M app.py` (từ nhiệm vụ trước), `?? reports/`, `?? tests/`; không có staging/commit/push.
- Hash dataset, artifact, `src/inference.py` không đổi so với trước thực nghiệm.
- Không có file nào trong `data/raw/`, `models/`, `src/` bị ghi.

### 11.4. Ghi chú về môi trường

- Python 3.11.9; scikit-learn 1.9.0; scipy 1.17.1.
- Không chạy notebook end-to-end trong môi trường này; mọi số liệu trong báo cáo đến từ script độc lập tái lập đúng A1.
