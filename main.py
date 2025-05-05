import os
import pandas as pd
import numpy as np
from scipy.stats import kurtosis, skew
import matplotlib.pyplot as plt
import re
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

# Định nghĩa thư mục chứa dữ liệu
DATA_DIR = "data/"  # Thay bằng đường dẫn thư mục chứa các tệp CSV/XLSX

# Danh sách tên tệp được cung cấp, với các tệp twist là .xlsx
FILE_NAMES = [
    "crack fault state-Vw=4.5.csv", "crack fault-Vw=4.csv", "Crack Fault-Vw=5.4.csv",
    "Crack Fault-Vw=5.csv", "Crack State-Vw=1.3.csv", "Crack state-Vw=2.8.csv",
    "Crack state-Vw=3.3.csv", "Erosion fault state-Vw=1.3.csv", "Erosion fault state-Vw=2.1.csv",
    "Erosion fault state-Vw=2.8.csv", "Erosion Fault state-Vw=3.4.csv", "Erosion Fault State-Vw=4.2.csv",
    "Erosion fault state-Vw=5.3.csv", "Erosion Fault state-Vw=5.csv", "H for Vw=2.3.csv",
    "H-for Vw=1.3.csv", "H-for-Vw=3.2.csv", "H-for-Vw=3.7.csv", "H-for-Vw=4.5.csv",
    "H-for-Vw=5.csv", "H-Vw=5.3.csv", "twist fault when Vw=1.3.xlsx", "twist fault when Vw=4.7.xlsx",
    "twist fault when Vw=5.3.xlsx", "twist fault when Vw=5.xlsx", "twist fault when Vwind=4.xlsx",
    "twist faultwhenVw=3.2.xlsx", "twsist faut when Vw=2.xlsx", "unbalance fault state=Vw=2.3.csv",
    "unbalance fault state-Vw=3.4.csv", "unbalance fault state-Vw=4.2.csv",
    "unbalance fault state-Vw=4.7.csv", "unbalance fault state-Vw=5.csv",
    "UnbalanceState-Vw=1.3.csv", "unbalnce fault state-Vw=3.csv"
]

def extract_label_and_vw(filename):
    """
    Suy ra nhãn và tốc độ gió (Vw) từ tên tệp.
    Trả về: (label, Vw)
    """
    filename_lower = filename.lower()
    if "healthy" in filename_lower or filename_lower.startswith("h"):
        label = "Healthy"
    elif "crack" in filename_lower:
        label = "Crack"
    elif "erosion" in filename_lower:
        label = "Erosion"
    elif "twist" in filename_lower or "twsist" in filename_lower:
        label = "Twist"
    elif "unbalance" in filename_lower or "unbalnce" in filename_lower:
        label = "Unbalance"
    else:
        label = "Unknown"
    
    # Trích xuất tốc độ gió (Vw hoặc Vwind) với biểu thức chính quy
    vw_match = re.search(r"(?:vw|vwind)=(\d+\.\d+|\d+)", filename_lower)
    vw = float(vw_match.group(1)) if vw_match else None
    if vw is None:
        print(f"Không thể trích xuất Vw từ tên tệp: {filename}")
    else:
        # Chuẩn hóa định dạng Vw thành số thập phân đầy đủ (ví dụ: 4.0 thay vì 4.)
        vw = float(f"{vw:.1f}")
    
    return label, vw

def augment_data(vibration, num_augmentations=2, noise_factor=0.01):
    """
    Tăng cường dữ liệu bằng cách thêm nhiễu vào tín hiệu rung động.
    vibration: Tín hiệu rung động gốc.
    num_augmentations: Số lượng tín hiệu mới được tạo.
    noise_factor: Độ lớn của nhiễu.
    Trả về: Danh sách các tín hiệu đã được tăng cường (bao gồm cả tín hiệu gốc).
    """
    augmented_signals = [vibration]  # Bao gồm tín hiệu gốc
    for _ in range(num_augmentations):
        noise = np.random.normal(0, noise_factor * np.std(vibration), len(vibration))
        augmented_signal = vibration + noise
        augmented_signals.append(augmented_signal)
    return augmented_signals

def read_and_analyze_data(data_dir, file_names, augment=True, num_augmentations=2):
    """
    Đọc và phân tích các tệp CSV/XLSX, với tùy chọn tăng cường dữ liệu.
    augment: Có thực hiện tăng cường dữ liệu hay không.
    num_augmentations: Số lượng tín hiệu mới được tạo từ mỗi tín hiệu gốc.
    """
    data_summary = []
    
    for filename in file_names:
        # Suy ra nhãn và tốc độ gió
        label, vw = extract_label_and_vw(filename)
        
        # Đọc tệp CSV hoặc XLSX
        filepath = os.path.join(data_dir, filename)
        try:
            if filename.endswith('.csv'):
                with open(filepath, 'r') as file:
                    lines = file.readlines()
                data = [line.strip().split(';') for line in lines[1:]]
            elif filename.endswith('.xlsx'):
                df_excel = pd.read_excel(filepath, engine='openpyxl')
                data = [[str(df_excel.iloc[i, 0]), str(df_excel.iloc[i, 1])] for i in range(len(df_excel))]
            else:
                print(f"Lỗi: Định dạng tệp không hỗ trợ - {filename}")
                continue
            
            # Lọc bỏ các dòng có giá trị rỗng hoặc không hợp lệ
            valid_data = []
            for row in data:
                if len(row) >= 2 and row[0].strip() and row[1].strip():  # Đảm bảo có ít nhất 2 cột
                    try:
                        float(row[0])  # Kiểm tra "Time - Voltage_1"
                        float(row[1])  # Kiểm tra "Amplitude - Voltage_1"
                        valid_data.append(row)
                    except ValueError:
                        print(f"Dòng lỗi trong {filename}: {row}")
                        continue
            if not valid_data:
                print(f"Không có dữ liệu hợp lệ trong tệp: {filename}")
                continue
            
            df = pd.DataFrame(valid_data, columns=['Time - Voltage_1', 'Amplitude - Voltage_1'])
            df['Time - Voltage_1'] = df['Time - Voltage_1'].astype(float)
            df['Amplitude - Voltage_1'] = df['Amplitude - Voltage_1'].astype(float)
        except FileNotFoundError:
            print(f"Lỗi: Tệp {filename} không tồn tại trong {data_dir}")
            continue
        except Exception as e:
            print(f"Lỗi khi đọc tệp {filename}: {e}")
            continue
        
        # Kiểm tra số mẫu
        num_rows = len(df)
        if num_rows != 500:
            print(f"Tệp {filename} không có đúng 500 mẫu: {num_rows}")
        
        # Lấy dữ liệu rung động từ cột "Amplitude - Voltage_1"
        vibration = df['Amplitude - Voltage_1'].values
        
        # Tăng cường dữ liệu (nếu bật tùy chọn augment)
        if augment:
            vibration_signals = augment_data(vibration, num_augmentations=num_augmentations)
        else:
            vibration_signals = [vibration]
        
        # Tính đặc trưng cho từng tín hiệu
        for i, vib_signal in enumerate(vibration_signals):
            # Tính toán đặc trưng thống kê
            rms = np.sqrt(np.mean(vib_signal**2))
            kurt = kurtosis(vib_signal, fisher=False)
            skw = skew(vib_signal)
            peak_to_peak = np.ptp(vib_signal)
            mean_val = np.mean(vib_signal)
            std_val = np.std(vib_signal)
            
            # Lưu thông tin
            data_summary.append({
                "Filename": f"{filename}_aug_{i}" if i > 0 else filename,
                "Label": label,
                "Wind_Speed_Vw": vw,
                "Num_Samples": len(vib_signal),
                "Columns": ['Time - Voltage_1', 'Amplitude - Voltage_1'],
                "RMS": rms,
                "Kurtosis": kurt,
                "Skewness": skw,
                "Peak_to_Peak": peak_to_peak,
                "Mean": mean_val,
                "Std": std_val
            })
        
        # Vẽ biểu đồ cho tín hiệu gốc
        if len(data_summary) <= num_augmentations + 1:
            plt.figure(figsize=(10, 4))
            plt.plot(df['Time - Voltage_1'], vibration, label=f"Signal (File: {filename})")
            plt.title("Vibration Signal Example")
            plt.xlabel("Time - Voltage_1")
            plt.ylabel("Amplitude - Voltage_1")
            plt.legend()
            plt.savefig('vibration_example.png')
            plt.close()
    
    return data_summary

def train_and_evaluate_model(summary_df):
    """
    Huấn luyện và đánh giá mô hình KNN.
    """
    features = ['RMS', 'Kurtosis', 'Skewness', 'Peak_to_Peak', 'Mean', 'Std', 'Wind_Speed_Vw']
    X = summary_df[features]
    y = summary_df['Label']
    
    # Chuẩn hóa dữ liệu
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Chia dữ liệu
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
    
    # Huấn luyện KNN
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train, y_train)
    
    # Đánh giá
    y_pred = model.predict(X_test)
    print("\nĐánh giá mô hình KNN (k=3):")
    print(classification_report(y_test, y_pred))
    print("Ma trận nhầm lẫn:")
    print(confusion_matrix(y_test, y_pred))
    
    return model, scaler

def predict_fault(model, scaler, new_data):
    """
    Dự đoán lỗi cho dữ liệu mới.
    """
    features = ['RMS', 'Kurtosis', 'Skewness', 'Peak_to_Peak', 'Mean', 'Std', 'Wind_Speed_Vw']
    X_new = new_data[features]
    X_new_scaled = scaler.transform(X_new)
    prediction = model.predict(X_new_scaled)
    return prediction

def main():
    # Đọc và phân tích dữ liệu (với tăng cường dữ liệu)
    data_summary = read_and_analyze_data(DATA_DIR, FILE_NAMES, augment=True, num_augmentations=2)
    
    if not data_summary:
        print("Không tìm thấy dữ liệu hợp lệ trong thư mục:", DATA_DIR)
        return
    
    # Chuyển thành DataFrame
    summary_df = pd.DataFrame(data_summary)
    
    # In thông tin tổng quan
    print(f"Tổng số mẫu sau tăng cường: {len(data_summary)}")
    print(f"Nhãn tìm thấy: {summary_df['Label'].unique()}")
    print(f"Tốc độ gió (Vw) tìm thấy: {summary_df['Wind_Speed_Vw'].unique()}")
    print(f"Số mẫu trung bình mỗi tệp: {summary_df['Num_Samples'].mean():.2f}")
    print("\nThông tin đặc trưng thống kê:")
    print(summary_df[['RMS', 'Kurtosis', 'Skewness', 'Peak_to_Peak', 'Mean', 'Std']].describe())
    
    # Lưu thông tin
    summary_df.to_csv('data_summary.csv', index=False)
    print("Đã lưu thông tin phân tích vào 'data_summary.csv'")
    
    # Vẽ biểu đồ
    plt.figure(figsize=(8, 6))
    for label in summary_df['Label'].unique():
        subset = summary_df[summary_df['Label'] == label]
        plt.scatter(subset['Wind_Speed_Vw'], subset['RMS'], label=label, alpha=0.6)
    plt.title("RMS Distribution by Label and Wind Speed")
    plt.xlabel("Wind Speed (Vw, m/s)")
    plt.ylabel("RMS Value")
    plt.legend()
    plt.savefig('rms_distribution.png')
    plt.close()
    
    # Huấn luyện và đánh giá mô hình
    model, scaler = train_and_evaluate_model(summary_df)

if __name__ == "__main__":
    main()
