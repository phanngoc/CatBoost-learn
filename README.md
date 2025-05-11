# Flight Delays Prediction with CatBoost

## English Description

This repository contains a machine learning project focused on predicting flight delays using the CatBoost gradient boosting algorithm. The project utilizes flight delay data from Fall 2018 to build and evaluate predictive models.

### Repository Structure

- **catbooster-starter.ipynb**: Jupyter notebook containing the main code for data processing, model training, and evaluation
- **data/**: Directory containing the dataset files
  - `flight_delays_train.csv`: Training dataset
  - `flight_delays_test.csv`: Test dataset
  - `sample_submission.csv.zip`: Sample submission file for Kaggle
- **catboost_info/**: Directory containing model training logs and information
- **kien-thuc/**: Directory containing educational materials and notes on machine learning concepts

### Getting Started

1. Clone this repository
2. Install the required dependencies:
   ```
   pip install catboost pandas numpy matplotlib scikit-learn jupyter
   ```
3. Open the `catbooster-starter.ipynb` notebook using Jupyter:
   ```
   jupyter notebook catbooster-starter.ipynb
   ```
4. Follow the instructions in the notebook to train the model and make predictions

### Dataset Description

The dataset contains historical flight information including departure/arrival details, airlines, and delay information. The goal is to predict whether a flight will be delayed or not.

## Mô tả Tiếng Việt

Kho lưu trữ này chứa một dự án học máy tập trung vào việc dự đoán chậm trễ chuyến bay sử dụng thuật toán CatBoost gradient boosting. Dự án sử dụng dữ liệu chậm trễ chuyến bay từ mùa Thu 2018 để xây dựng và đánh giá các mô hình dự đoán.

### Cấu trúc kho lưu trữ

- **catbooster-starter.ipynb**: Notebook Jupyter chứa mã chính cho việc xử lý dữ liệu, huấn luyện mô hình, và đánh giá
- **data/**: Thư mục chứa các tệp dữ liệu
  - `flight_delays_train.csv`: Tập dữ liệu huấn luyện
  - `flight_delays_test.csv`: Tập dữ liệu kiểm tra
  - `sample_submission.csv.zip`: Tệp mẫu nộp cho Kaggle
- **catboost_info/**: Thư mục chứa nhật ký huấn luyện mô hình và thông tin
- **kien-thuc/**: Thư mục chứa tài liệu giáo dục và ghi chú về các khái niệm học máy

### Bắt đầu

1. Clone kho lưu trữ này
2. Cài đặt các thư viện cần thiết:
   ```
   pip install catboost pandas numpy matplotlib scikit-learn jupyter
   ```
3. Mở notebook `catbooster-starter.ipynb` bằng Jupyter:
   ```
   jupyter notebook catbooster-starter.ipynb
   ```
4. Làm theo hướng dẫn trong notebook để huấn luyện mô hình và đưa ra dự đoán

### Mô tả Dữ liệu

Tập dữ liệu chứa thông tin chuyến bay lịch sử bao gồm chi tiết khởi hành/đến, hãng hàng không, và thông tin về sự chậm trễ. Mục tiêu là dự đoán liệu một chuyến bay có bị chậm trễ hay không.