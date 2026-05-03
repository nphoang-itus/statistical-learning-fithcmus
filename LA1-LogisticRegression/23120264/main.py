import json
import numpy as np
from map_feature import map_feature

def load_config(config_file='config.json'):
  """Đọc các siêu tham số từ file config.json"""
  with open(config_file, 'r') as file:
      config = json.load(file)
  
  alpha = config['Alpha']
  lambda_ = config['Lambda'] # Dùng lambda_ vì lambda là từ khóa mặc định của Python
  num_iter = config['NumIter']
  
  return alpha, lambda_, num_iter

def load_data(data_file='training_data.txt'):
  """Đọc dữ liệu và phân tách thành X_raw và y"""
  # Sử dụng numpy để đọc file txt phân tách bằng dấu phẩy
  data = np.loadtxt(data_file, delimiter=',')
  
  # Tách X (cột 1, 2) và y (cột 3)
  X_raw = data[:, :2]  # Lấy tất cả các hàng, 2 cột đầu tiên
  y = data[:, 2]       # Lấy tất cả các hàng, cột thứ 3 (index 2)
  
  return X_raw, y

def sigmoid(z):
  """
  Tính giá trị hàm sigmoid: g(z) = 1 / (1 + e^-z)
  """
  # np.clip giúp giới hạn giá trị của z, tránh hiện tượng tràn số (overflow) 
  # khi hàm exp(-z) nhận giá trị z quá âm.
  z = np.clip(z, -250, 250)
  return 1.0 / (1.0 + np.exp(-z))

def compute_cost(X, y, theta, lambda_):
  """
  Tính chi phí J(theta) có điều chuẩn (regularized cost function)
  """
  m = len(y)
  
  # Tính dự đoán (xác suất)
  h = sigmoid(X.dot(theta))
  
  # Epsilon nhỏ để tránh lỗi toán học log(0) khi h quá gần 0 hoặc 1
  epsilon = 1e-5
  
  # Phần 1: Cross-entropy cost
  cost = (1 / m) * np.sum(-y * np.log(h + epsilon) - (1 - y) * np.log(1 - h + epsilon))
  
  # Phần 2: L2 Regularization term (Lưu ý: bỏ qua theta[0])
  reg_term = (lambda_ / (2 * m)) * np.sum(np.square(theta[1:]))
  
  J = cost + reg_term
  return J

def compute_gradient(X, y, theta, lambda_):
  """
  Tính vector gradient có điều chuẩn
  """
  m = len(y)
  
  # Tính dự đoán (xác suất)
  h = sigmoid(X.dot(theta))
  
  # Sai số giữa dự đoán và thực tế
  error = h - y
  
  # Tính gradient cơ bản (chưa có điều chuẩn)
  grad = (1 / m) * X.T.dot(error)
  
  # Khởi tạo vector điều chuẩn cùng kích thước với theta
  reg_term = (lambda_ / m) * theta
  
  # Không áp dụng điều chuẩn cho theta_0 (phần tử đầu tiên)
  reg_term[0] = 0 
  
  # Cộng gradient cơ bản với phần điều chuẩn
  grad = grad + reg_term
  
  return grad

def gradient_descent(X, y, theta, alpha, lambda_, num_iter):
  """
  Thực hiện thuật toán Gradient Descent để tối ưu hóa tham số theta
  """
  J_history = [] # Mảng lưu lại chi phí sau mỗi vòng lặp để theo dõi
  
  print("\n--- Starting Gradient Descent ---")
  for i in range(num_iter):
      # Bước 1: Tính vector gradient hiện tại
      grad = compute_gradient(X, y, theta, lambda_)
      
      # Bước 2: Cập nhật tham số theta
      theta = theta - alpha * grad
      
      # In tiến độ và lưu cost mỗi 1000 vòng lặp (giúp theo dõi mô hình học)
      if i % 1000 == 0 or i == num_iter - 1:
          cost = compute_cost(X, y, theta, lambda_)
          J_history.append(cost)
          print(f"Iteration {i:5d} | Cost: {cost:.6f}")
          
  print("--- Training Completed ---")
  return theta, J_history

def save_model(theta, filename='model.json'):
  """
  Lưu tham số mô hình vào tập tin JSON
  """
  # Hàm json.dump không nhận trực tiếp Numpy Array, 
  # nên ta cần chuyển theta thành Python list bằng hàm .tolist()
  model_data = {
      "theta": theta.tolist() 
  }
  with open(filename, 'w') as file:
      json.dump(model_data, file, indent=4)

def predict(X, theta, threshold=0.5):
    """
    Dự đoán nhãn (0 hoặc 1) dựa trên xác suất và ngưỡng (threshold)
    """
    # Tính xác suất bằng hàm sigmoid
    probabilities = sigmoid(X.dot(theta))
    
    # Biến đổi xác suất thành mảng [0, 1] dựa trên threshold
    # .astype(int) giúp chuyển giá trị True/False thành 1/0
    predictions = (probabilities >= threshold).astype(int)
    
    return predictions

def evaluate(y_true, y_pred):
  """
  Đánh giá mô hình dựa trên Accuracy, Precision, Recall, F1-score
  """
  # Tính toán các giá trị cơ sở TP, TN, FP, FN
  TP = np.sum((y_true == 1) & (y_pred == 1))
  TN = np.sum((y_true == 0) & (y_pred == 0))
  FP = np.sum((y_true == 0) & (y_pred == 1))
  FN = np.sum((y_true == 1) & (y_pred == 0))
  
  # Tránh lỗi chia cho 0 (ZeroDivisionError)
  accuracy = (TP + TN) / len(y_true) if len(y_true) > 0 else 0
  precision = TP / (TP + FP) if (TP + FP) > 0 else 0
  recall = TP / (TP + FN) if (TP + FN) > 0 else 0
  f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
  
  # Đóng gói kết quả vào dictionary
  report = {
      "accuracy": accuracy,
      "precision": precision,
      "recall": recall,
      "f1_score": f1_score
  }
  return report

def save_report(report, filename='classification_report.json'):
  """
  Lưu báo cáo đánh giá vào tập tin JSON
  """
  with open(filename, 'w') as file:
      json.dump(report, file, indent=4)

if __name__ == "__main__":
  # 1. Load configuration
  alpha, lambda_, num_iter = load_config()
  print("--- Training Configuration ---")
  print(f"Alpha: {alpha}, Lambda: {lambda_}, NumIter: {num_iter}\n")

  # 2. Load raw data
  X_raw, y = load_data()
  print("--- Raw Data ---")
  print(f"X_raw shape: {X_raw.shape}")
  print(f"y shape: {y.shape}")
  print(f"First 5 samples of X_raw:\n{X_raw[:5]}\n")

  # 3. Feature Mapping (Tiền xử lý dữ liệu)
  # Tách riêng cột x1 và x2 từ X_raw để đưa vào hàm map_feature
  x1 = X_raw[:, 0]
  x2 = X_raw[:, 1]
  
  # Ánh xạ dữ liệu sang 28 chiều
  X = map_feature(x1, x2)
  
  print("--- Feature Mapping ---")
  print(f"X shape after mapping: {X.shape}")
  print(f"First sample of X:\n{X[0]}")

  # # 4.1 Kiểm tra các hàm toán học
  # # Lấy số lượng đặc trưng n (số cột của X) sau khi map_feature
  # n_features = X.shape[1] 
  
  # # Khởi tạo tham số theta ban đầu bằng 0
  # initial_theta = np.zeros(n_features)
  
  # # Tính chi phí và gradient ban đầu
  # initial_cost = compute_cost(X, y, initial_theta, lambda_)
  # initial_grad = compute_gradient(X, y, initial_theta, lambda_)
  
  # print("\n--- Testing Core Functions ---")
  # print(f"Initial Cost (with theta zeros): {initial_cost:.4f}") # Thường khoảng 0.6931
  # print(f"First 5 elements of Initial Gradient:\n{initial_grad[:5]}")

  # 4.2 Model Training (Huấn luyện mô hình)
  # Khởi tạo tham số theta ban đầu bằng 0
  n_features = X.shape[1] 
  initial_theta = np.zeros(n_features)
  
  # Chạy Gradient Descent
  optimal_theta, J_history = gradient_descent(
      X=X, 
      y=y, 
      theta=initial_theta, 
      alpha=alpha, 
      lambda_=lambda_, 
      num_iter=num_iter
  )
  
  # 5. Lưu mô hình đã huấn luyện
  save_model(optimal_theta, filename='model.json')
  
  print("\n--- Model Parameters Saved ---")
  print("Check your folder for 'model.json' file.")

  # 6. Predict and Evaluate (Dự đoán và Đánh giá)
  print("\n--- Model Evaluation ---")
  
  # Thực hiện dự đoán trên toàn bộ tập dữ liệu huấn luyện
  y_pred = predict(X, optimal_theta)
  
  # Đánh giá kết quả
  classification_report = evaluate(y, y_pred)
  
  # In báo cáo ra màn hình
  print(f"Accuracy : {classification_report['accuracy'] * 100:.2f}%")
  print(f"Precision: {classification_report['precision'] * 100:.2f}%")
  print(f"Recall   : {classification_report['recall'] * 100:.2f}%")
  print(f"F1-Score : {classification_report['f1_score'] * 100:.2f}%")
  
  # Lưu báo cáo vào file JSON
  save_report(classification_report, filename='classification_report.json')
  print("\n--- Report Saved ---")
  print("Check your folder for 'classification_report.json' file.")