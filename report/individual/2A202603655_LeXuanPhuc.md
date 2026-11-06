# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

- **MSSV:** 2A202603655
- **Họ và tên:** Lê Xuân Phúc
- **SEED:** 2711

## 1. Bằng chứng

| Tệp | Có? |
| --- | :---: |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ☑ |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ☑ |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | ☑ |

## 2. Kết quả chính

| Phương pháp tiếp cận | F1 (test) | F1 (SUB 40) | p95 ms/tin | USD / 1k tin |
| :--- | :---: | :---: | :---: | :---: |
| **1. BoW + LogReg (tự code)** | 0.633 | 0.722 | 0.147 | $0.000 |
| **2. BoW + MLP (deep learning)** | 0.949 | 0.947 | 2.479 | $0.000 |
| **3a. Encoder (MiniLM) + kNN** | 0.923 | 1.000 | 65.487 | $0.000 |
| **3b. Encoder (MiniLM) + LogReg** | 0.380 | 0.826 | 51.404 | $0.000 |
| **3c. Encoder (MiniLM) + MLP head** | **0.963** | **1.000** | **35.385** | **$0.000** |
| **4a. LLM nhỏ zero-shot (`gpt-6-luna`)** | — | 0.000 | 1869.382 | $0.006 |
| **4b. LLM lớn zero-shot (`gpt-6.1-terra`)** | — | 0.000 | 1383.726 | $0.057 |
| **4c. LLM nhỏ few-shot (`gpt-6-luna`)** | — | 0.000 | 1223.543 | $0.016 |

## 3. Dự đoán và thực tế

1. **`llm_models` (Thí nghiệm 4.A):**
   * *Đoán:* F1 của model lớn cao hơn khoảng 0.05 hoặc tương đương, F1 đạt ~0.95 ngang ngửa mô hình 3b.
   * *Thực tế:* F1 = 0.000 (invalid = 40/40).
   * *Vì sao lệch:* Khi đặt `max_tokens=3`, phản hồi của LLM bị cắt cụt hoặc chứa reasoning tokens/khoảng trắng thừa, khiến regex không parse được nhãn dẫn đến toàn bộ tin bị fallback về 0.
2. **`ml_f1` (Thí nghiệm 1.A):**
   * *Đoán:* F1 đạt khoảng 0.80.
   * *Thực tế:* F1 test thực tế chỉ đạt 0.633.
   * *Vì sao lệch:* BoW nhị phân 0/1 bị hạn chế nghiêm trọng trước các từ đồng nghĩa và từ ngữ viết tắt trong tin nhắn spam thực tế, khiến chỉ số Recall bị kéo tụt xuống 0.609.
3. **`ml_lr` (Thí nghiệm 1.B):**
   * *Đoán:* lr=0.001 học chậm, lr=50 loss dao động và nổ phân kỳ.
   * *Thực tế:* Khớp hoàn toàn. Tại lr=50, loss vọt lên ~14.8 ngay vòng đầu do hiện tượng overshooting trước khi cố gắng hồi tụ lại quanh 0.279.

## 4. Phân tích 50 case khó

* **Model tụt nhiều nhất:** Mô hình **BoW + LogReg (Phần 1)** tụt mạnh nhất (accuracy chỉ còn ~45%), tụt nặng nề nhất ở 2 nhóm: `obfuscation` (lách ký tự) và `ham_looks_spam` (tin thường có từ khóa nhạy cảm).
* **2 case cụ thể & Cơ chế sai:**
  1. *Case Obfuscation:* Tin spam chứa số điện thoại quảng cáo hoặc chữ biến dị (`0871750.77.11!`, `w!n`, `fr33`). 
     * *Cơ chế:* Bộ tách từ của BoW bỏ qua token lạ vì không có trong từ điển `vocab`, làm vector câu toàn số 0 $\to$ BoW đoán nhầm thành Ham.
  2. *Case Ham chứa từ nhạy cảm:* Tin hẹn hò *"I am free to meet you at noon"*.
     * *Cơ chế:* BoW gán trọng số dương rất lớn cho từ `free` (vốn hay nằm trong spam trúng thưởng), bỏ qua ngữ cảnh "rảnh rỗi" $\to$ BoW chặn nhầm thành Spam (False Positive).

## 5. Thử thách cá nhân

* **Đề bài (#6 theo SEED 2711):** Sửa prompt để LLM trả về JSON `{"label": ..., "reason": ...}`, viết hàm parse JSON, chạy trên 10 tin SUB và so sánh token/chi phí với bản 1 từ.
* **Cách làm:** Định nghĩa prompt yêu cầu trả JSON chuẩn, dùng regex `re.search` và `json.loads` trích xuất trường `label` để phân loại.
* **Kết quả:** Chi phí output tăng gấp ~30–40 lần (output từ 1 token tăng lên ~35–45 tokens), độ trễ latency tăng tỷ lệ thuận theo số lượng token sinh ra.
* **Giải thích:** Với việc lọc tin nhắn thời gian thực quy mô lớn thì **không đáng** vì quá chậm và đắt; nhưng với hệ thống kiểm duyệt cần giải thích quyết định (Explainable AI) cho người dùng thì **rất đáng**.

## 6. Phương án cho sản phẩm thật

* **Yêu cầu:** Lọc 10 triệu SMS/ngày (~116 tin/s), yêu cầu **p95 < 50 ms/tin**.
* **Phương án lựa chọn:** Chọn **`3c. Encoder (MiniLM) + MLP head`** (hoặc kiến trúc lai với `2. BoW + MLP`).
* **Dẫn chứng số liệu:**
  * Độ trễ p95 đạt **35.385 ms/tin** (thỏa mãn tiêu chuẩn < 50 ms).
  * Thông lượng đạt **52.7 tin/s** trên 1 lõi (khi scale đa luồng dễ dàng vượt mốc 116 tin/s).
  * Chất lượng F1 đạt **0.963** (cao nhất toàn bộ các phương pháp).
  * Chi phí API: **$0 USD** (chạy local trên hạ tầng tự host, không phụ thuộc bên thứ 3).
* **Chấp nhận đánh đổi:** Loại bỏ hoàn toàn phương án thuần LLM vì LLM có độ trễ p95 > 1.2s (chậm gấp 25 lần chuẩn SLA) và tiêu tốn hàng ngàn USD tiền API mỗi tháng.

## 7. AI log và điều đã học

| STT | Bạn hỏi AI gì (Khái niệm / Phân tích) | AI giải thích nội dung gì | Bạn đã kiểm chứng / hiểu ra gì |
|:---:|---|---|---|
| 1 | Data leakage là gì và ảnh hưởng của tin nhắn trùng lặp? | Giải thích việc dữ liệu test xuất hiện trong train làm mô hình học vẹt, thổi phồng điểm số giả tạo | Hiểu vì sao bắt buộc phải lọc trùng lặp trước khi chia dữ liệu |
| 2 | Ý nghĩa của mô hình lười (Baseline Accuracy) | Phân tích tỉ lệ 4516/5169 = 87.37% khi dữ liệu mất cân bằng | Nhận ra accuracy 87% là vô nghĩa nếu chỉ đoán toàn bộ là ham |
| 3 | Ý nghĩa bản chất ma trận nhầm lẫn TP, FP, FN, TN và P/R | Phân biệt báo động giả (FP - chặn nhầm) và bỏ lọt (FN - rò rỉ spam) | Hiểu vì sao bài toán SMS ưu tiên F1 và Precision cao hơn Accuracy |
| 4 | Ảnh hưởng của cập nhật trọng số Gradient Descent với từ "free" | Giải thích khi đoán thiếu spam thì error < 0, trừ đi số âm làm tăng trọng số w của từ "free" | Nắm rõ bản chất toán học của thuật toán hạ độ dốc |
| 5 | Vì sao Bag-of-Words thất bại ở các câu chứa từ đa nghĩa? | Phân tích BoW chỉ đếm từ rời rạc, không phân biệt được "free" rảnh rỗi vs "free" miễn phí | Hiểu được động lực chuyển sang mô hình biểu diễn ngữ nghĩa (Embedding) |
| 6 | Đọc đồ thị Loss theo Epoch và hiện tượng ở lr=50 | Giải thích hiện tượng overshooting: bước nhảy quá to làm văng khỏi điểm cực tiểu | Nhận diện được đường cong learning curve tối ưu vs phân kỳ |

* **2–3 câu đúc kết điều hiểu ra hôm nay:**
  1. Độ chính xác (Accuracy) là con số dễ gây ảo tưởng nhất khi dữ liệu mất cân bằng; muốn đánh giá chuẩn hệ thống AI bắt buộc phải nhìn vào F1, Precision và Recall.
  2. Mô hình phức tạp nhiều tham số (như Deep Learning hay LLM) không phải lúc nào cũng là giải pháp tốt nhất trong sản xuất; việc biểu diễn đúng ngữ nghĩa dữ liệu (Embedding) kết hợp bộ phân loại nhẹ mới là chìa khóa tối ưu cả về độ trễ, chất lượng và chi phí.
