
## **Bố Cục Hoàn Chỉnh cho Partitioning Methods – Clustering**

---

### **1. Giới Thiệu về Clustering**  
- **Định nghĩa**: Clustering là phương pháp phân chia dữ liệu thành các cụm để các đối tượng trong cùng cụm có sự tương đồng cao hơn so với đối tượng ở cụm khác.  
- **Vai trò**: Tìm kiếm mẫu hoặc cấu trúc tiềm ẩn trong dữ liệu không có nhãn.  
- **Ví dụ thực tế**: 
  - Phân nhóm khách hàng theo hành vi mua sắm.
  - Phân loại tài liệu hoặc tin tức thành các chủ đề.

---

### **2. Partitioning Methods – Định Nghĩa và Cơ Chế Hoạt Động**  
- **Partitioning Methods là gì?**  
  - Các phương pháp này phân vùng dữ liệu thành \(k\) cụm, với mỗi đối tượng thuộc về đúng một cụm.
  - Phù hợp cho dữ liệu vừa và nhỏ, nơi **số cụm \(k\)** được xác định trước.

- **Cơ chế hoạt động**:
  - Sử dụng các **centroid** hoặc điểm đại diện để phân cụm.
  - Mục tiêu: Tối ưu hóa khoảng cách giữa các điểm và centroid của chúng.

---

### **3. Thuật Toán K-means – Cơ Chế Chi Tiết**  
- **Ý tưởng**: K-means tối thiểu hóa **Sum of Squared Errors (SSE)** để đảm bảo các điểm gần với centroid của cụm.
  
- **Các bước thuật toán K-means**:
  1. **Khởi tạo**: Chọn ngẫu nhiên \(k\) centroid hoặc sử dụng **K-means++** để khởi tạo tốt hơn.
  2. **Gán cụm**: Mỗi điểm dữ liệu được gán vào cụm gần nhất theo **khoảng cách Euclidean**.
  3. **Cập nhật centroid**: Tính lại centroid dựa trên các điểm trong cụm.
  4. **Lặp lại**: Tiếp tục cho đến khi các centroid không thay đổi hoặc đạt số lần lặp tối đa.

- **Ví dụ minh họa**: Dùng tập dữ liệu tọa độ thành phố để phân thành các cụm địa lý gần nhau.

---

### **4. Hàm Mất Mát và Tối Ưu Hóa**  
- **Hàm mất mát**: 
  \[
  SSE = \sum_{j=1}^{k} \sum_{x_i \in C_j} ||x_i - \mu_j||^2
  \]
  - \(\mu_j\): Centroid của cụm \(C_j\).
  
- **Tối ưu hóa**: 
  - **Coordinate Descent**: Cập nhật lần lượt centroid và cụm cho đến khi hội tụ.
  - **K-means++** giúp chọn centroid tốt hơn, tránh local minima.

---

### **5. Ưu và Nhược Điểm của Partitioning Methods**  
- **Ưu điểm**:
  - Dễ hiểu, triển khai đơn giản.
  - Thời gian chạy nhanh với tập dữ liệu vừa phải.

- **Nhược điểm**:
  - Nhạy cảm với khởi tạo ban đầu và outliers.
  - Không phù hợp với dữ liệu có hình dạng phức tạp hoặc phi tuyến.

---

### **6. Các Kỹ Thuật Cải Tiến cho Partitioning Methods**  
- **K-means++**: Chọn centroid thông minh hơn để cải thiện hội tụ.
- **Mini-batch K-means**: Áp dụng trên các batch dữ liệu nhỏ để tăng tốc cho dữ liệu lớn.
- **GMM (Gaussian Mixture Models)**: Xử lý cụm chồng lấn bằng cách mô hình hóa các cụm như phân phối Gaussian.

---

### **7. Lựa Chọn Số Cụm \(k\) Thích Hợp**  
- **Elbow Method**: Chọn \(k\) tại điểm mà SSE bắt đầu giảm chậm.
- **Silhouette Score**:
  \[
  s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}
  \]
  - \(a(i)\): Khoảng cách trung bình từ \(i\) đến các điểm trong cùng cụm.
  - \(b(i)\): Khoảng cách trung bình từ \(i\) đến cụm gần nhất.

---

### **8. Ứng Dụng Thực Tiễn của Partitioning Methods**  
- **Phân nhóm khách hàng**: Xây dựng các chiến dịch marketing cá nhân hóa.
- **Hệ thống gợi ý sản phẩm**: Đề xuất sản phẩm dựa trên cụm hành vi mua sắm.
- **Nhận diện bất thường**: Phát hiện gian lận tài chính qua phân tích outliers.
- **Phân tích văn bản**: Phân cụm các bài báo hoặc tài liệu thành các chủ đề.

---

### **9. Các Thách Thức và Hạn Chế**  
- **Curse of Dimensionality**: Khi số chiều tăng, các điểm dữ liệu trở nên khó phân biệt.
  - **Giải pháp**: Áp dụng giảm chiều như **PCA** hoặc **t-SNE**.

- **Outliers và Khởi tạo kém**: Làm sai lệch centroid và ảnh hưởng đến kết quả.
  - **Giải pháp**: Dùng **K-means++** hoặc K-medoids.

---

### **10. Partitioning Methods – Các Thuật Toán Mở Rộng**  

#### **10.1 K-medoids Clustering**  
- **Ý tưởng**: Sử dụng **medoid** làm đại diện thay vì centroid.
- **Ưu điểm**: Giảm ảnh hưởng của outliers.
- **Nhược điểm**: Chạy chậm hơn so với K-means.

#### **10.2 BFR (Bradley-Fayyad-Reina)**  
- **Ý tưởng**: Mở rộng K-means cho dữ liệu lớn, với việc tóm tắt dữ liệu bằng **Discard Set, Compressed Set, Retained Set**.
- **Ưu điểm**: Phù hợp với dữ liệu không thể xử lý trong bộ nhớ.
- **Nhược điểm**: Giả định các cụm có phân phối Gaussian.

#### **10.3 CURE (Clustering Using Representatives)**  
- **Ý tưởng**: Chọn nhiều điểm đại diện để mô tả cụm, dịch chuyển về phía centroid.
- **Ưu điểm**: Xử lý tốt dữ liệu có hình dạng cụm phức tạp và nhiều outliers.
- **Nhược điểm**: Chi phí tính toán cao hơn.

---

## **11. Tổng Kết**  
- **Partitioning Methods** cung cấp các công cụ mạnh mẽ và hiệu quả để phân tích dữ liệu, với mỗi thuật toán có ưu và nhược điểm riêng.
- Lựa chọn thuật toán cần dựa trên đặc điểm dữ liệu và yêu cầu cụ thể của bài toán.
