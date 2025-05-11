# Cách thuật toán **Denoising Diffusion Probabilistic Models (DDPM)** hoạt động.

### 1. **Khái Niệm Về Diffusion Process**

**Diffusion Process** là một quá trình ngẫu nhiên mô tả sự lan tỏa của một đại lượng (như hạt, thông tin, hoặc dữ liệu) qua thời gian. Trong bối cảnh xử lý hình ảnh, quá trình này được sử dụng để thêm nhiễu vào ảnh, sau đó học một mô hình ngược để khôi phục ảnh sạch từ ảnh nhiễu. 

- **Forward Process** (Quá trình tiến): Là quá trình thêm nhiễu vào ảnh gốc qua các bước thời gian, ta có công thức sau:

$$x_t = \sqrt{1 - \beta_t} \cdot x_{t-1} + \sqrt{\beta_t} \cdot \epsilon_t$$

- **Reverse Process** (Quá trình ngược): Là quá trình học để tái tạo ảnh sạch từ ảnh nhiễu qua các bước, thuật toán sẽ học cách đảo ngược quá trình thêm nhiễu này. Công thức đảo ngược của quá trình tiến được biểu diễn như sau:

$$p_\theta(x_{t-1} | x_t) = \mathcal{N}(x_{t-1}; \hat{\mu}_\theta(x_t, t), \hat{\beta}_t I)$$

Ở đây:
- $x_t$ là ảnh nhiễu tại bước thời gian $t$,
- $\beta_t$ là mức độ nhiễu tại bước $t$,
- $\epsilon_t$ là nhiễu ngẫu nhiên (sampled from a normal distribution),
- $\hat{\mu}_\theta(x_t, t)$ là trung bình dự đoán của mô hình (được học từ mạng neural),
- $\hat{\beta}_t$ là phương sai dự đoán từ mô hình.

### 2. **Thuật Toán Bổ Trợ Giúp Hiểu Diffusion**

Để hiểu và cải thiện quá trình **Diffusion**, có một số thuật toán bổ trợ giúp tăng cường hiệu quả:

#### a. **Noise Scheduling (Lịch Trình Nhiễu)**
Để mô phỏng quá trình tiến, cần một **lịch trình nhiễu** $\beta_1, \beta_2, ..., \beta_T$, xác định lượng nhiễu thêm vào mỗi bước thời gian. Lịch trình này có thể được tối ưu để đảm bảo rằng quá trình lan truyền sẽ đạt được kết quả tốt nhất.

- **Linear Schedule**: Một cách đơn giản là sử dụng $\beta_t = \frac{t}{T}$, tức là tăng dần mức độ nhiễu theo thời gian.
  
- **Cosine Schedule**: Mức độ nhiễu có thể được điều chỉnh theo cách hàm cosine, giúp tránh việc thêm quá nhiều nhiễu vào những bước đầu, giữ được thông tin gốc tốt hơn trong giai đoạn đầu.

#### b. **Thuật Toán Mạng Neural Dự Đoán Nhiễu**
**Mạng neural $\epsilon_{\theta}(x_t, t)$** là một phần quan trọng trong việc học mô phỏng ngược quá trình diffusion. Mạng này học cách dự đoán nhiễu từ ảnh nhiễu $x_t$ tại bước thời gian $t$. Mục tiêu là dự đoán chính xác $\epsilon_t$ sao cho quá trình phục hồi ảnh sạch đạt được kết quả tốt nhất.

- **Mạng Neural U-Net**: Đây là một kiến trúc mạng neural phổ biến trong các mô hình như DDPM. U-Net có cấu trúc đặc biệt với các khối mã hóa và giải mã (encoder-decoder), cho phép kết hợp thông tin từ nhiều cấp độ khác nhau trong quá trình học.

- **Thuật Toán Loss Function**: Mạng neural được huấn luyện với hàm **loss** dựa trên sự khác biệt giữa nhiễu dự đoán và nhiễu thực tế. Loss function phổ biến là **Mean Squared Error (MSE)**:

$$\mathcal{L}(\hat{\epsilon}_t, \epsilon_t) = \frac{1}{N} \sum_{i=1}^N (\hat{\epsilon}_{t,i} - \epsilon_{t,i})^2$$

Với $\hat{\epsilon}_t$ là dự đoán nhiễu từ mạng neural và $\epsilon_t$ là nhiễu thực tế.

#### c. **Reverse Process (Quá Trình Ngược)**
Một khi mạng đã học được cách dự đoán nhiễu tại các bước thời gian $t$, quá trình ngược bắt đầu từ ảnh nhiễu tại thời điểm cuối cùng $x_T$ (khi nhiễu cực lớn) và sử dụng mạng neural để khôi phục ảnh sạch dần theo từng bước thời gian $t = T, T-1, ..., 1$.

Mỗi bước trong quá trình ngược này có thể được mô tả bằng công thức:

$$x_{t-1} = \hat{\mu}_\theta(x_t, t) + \hat{\beta}_t \cdot \hat{\epsilon}_t$$

### 3. **Cải Tiến và Tối Ưu Diffusion Models**

Các cải tiến và tối ưu hóa cho mô hình diffusion giúp tăng cường hiệu quả và chất lượng ảnh phục hồi:

- **Classifier-Free Guidance**: Đây là một phương pháp để cải thiện chất lượng của mô hình diffusion bằng cách không yêu cầu phân loại ảnh trong quá trình huấn luyện, mà thay vào đó là học cách phát hiện và phục hồi ảnh sạch một cách hiệu quả.

- **Improved Sampling Methods**: Các phương pháp sampling mới như **Laplacian Sampling** hoặc **Non-Linear Sampling** có thể giúp giảm thiểu nhiễu trong quá trình sinh ảnh và giúp phục hồi hình ảnh chất lượng cao hơn.

### 4. **Một Ví Dụ Cụ Thể Với Dữ Liệu**

Giả sử chúng ta có một ảnh sạch $x_0 = [0.8, 0.5, 0.3]$ và nhiễu tại bước $t = 1$ là $\epsilon_1 = [0.2, -0.1, 0.1]$, với mức độ nhiễu $\beta_1 = 0.1$. 

Quá trình thêm nhiễu vào ảnh sạch:

$$x_1 = \sqrt{1 - 0.1} \cdot [0.8, 0.5, 0.3] + \sqrt{0.1} \cdot [0.2, -0.1, 0.1]$$

Sau đó, mô hình neural $\epsilon_{\theta}(x_1, 1)$ sẽ học cách dự đoán nhiễu $\epsilon_1$ từ ảnh nhiễu $x_1$, sử dụng thuật toán MSE để tối ưu hóa mô hình.

### 5. **Tóm Tắt Các Thuật Toán Quan Trọng**

- **Forward Process**: Thêm nhiễu vào ảnh sạch qua các bước thời gian.
- **Reverse Process**: Dùng mạng neural để phục hồi ảnh sạch từ ảnh nhiễu.
- **Noise Scheduling**: Lập kế hoạch mức độ nhiễu tại mỗi bước thời gian để kiểm soát quá trình thêm nhiễu.
- **Loss Function**: Dựa trên MSE để huấn luyện mạng neural dự đoán nhiễu.
- **Classifier-Free Guidance** và **Improved Sampling**: Các cải tiến giúp tối ưu hóa quá trình sinh ảnh.

Qua các thuật toán và cải tiến này, mô hình Diffusion có thể tạo ra các ảnh chất lượng cao, giảm nhiễu hiệu quả và phục hồi thông tin chính xác từ các ảnh nhiễu.