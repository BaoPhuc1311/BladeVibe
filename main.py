import os
import pandas as pd
import numpy as np
from scipy.stats import kurtosis, skew
import matplotlib.pyplot as plt
import re

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

def read_and_analyze_data(data_dir, file_names):
    """
    Đọc và phân tích các tệp CSV/XLSX với định dạng 'time;amplitude' (CSV) hoặc 2 cột (XLSX).
    Trả về: Danh sách thông tin tệp và đặc trưng.
    """
    data_summary = []
    
    for filename in file_names:
        # Suy ra nhãn và tốc độ gió
        label, vw = extract_label_and_vw(filename)
        
        # Đọc tệp CSV hoặc XLSX
        filepath = os.path.join(data_dir, filename)
        try:
            if filename.endswith('.csv'):
                # Đọc dữ liệu thô và phân tách bằng dấu chấm phẩy
                with open(filepath, 'r') as file:
                    lines = file.readlines()
                data = [line.strip().split(';') for line in lines[1:]]  # Bỏ dòng tiêu đề
            elif filename.endswith('.xlsx'):
                # Đọc tệp XLSX với 2 cột sẵn có
                df_excel = pd.read_excel(filepath, engine='openpyxl')
                # Đảm bảo các cột được đặt tên đúng
                df_excel.columns = ['Time', 'Amplitude_Voltage_1']
                data = [[str(df_excel.iloc[i, 0]), str(df_excel.iloc[i, 1])] for i in range(len(df_excel))]
            else:
                print(f"Lỗi: Định dạng tệp không hỗ trợ - {filename}")
                continue
            
            # Lọc bỏ các dòng có giá trị rỗng hoặc không hợp lệ
            valid_data = []
            for row in data:
                if len(row) == 2 and row[0].strip() and row[1].strip():  # Đảm bảo có 2 cột và không rỗng
                    try:
                        float(row[0])  # Kiểm tra time
                        float(row[1])  # Kiểm tra amplitude
                        valid_data.append(row)
                    except ValueError:
                        print(f"Dòng lỗi trong {filename}: {row}")
                        continue
            if not valid_data:
                print(f"Không có dữ liệu hợp lệ trong tệp: {filename}")
                continue
            
            df = pd.DataFrame(valid_data, columns=['Time', 'Amplitude_Voltage_1'])
            df['Time'] = df['Time'].astype(float)
            df['Amplitude_Voltage_1'] = df['Amplitude_Voltage_1'].astype(float)
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
        
        # Lấy dữ liệu rung động
        vibration = df['Amplitude_Voltage_1'].values
        
        # Tính toán đặc trưng thống kê
        rms = np.sqrt(np.mean(vibration**2))
        kurt = kurtosis(vibration, fisher=False)
        skw = skew(vibration)
        peak_to_peak = np.ptp(vibration)
        mean_val = np.mean(vibration)
        std_val = np.std(vibration)
        
        # Lưu thông tin
        data_summary.append({
            "Filename": filename,
            "Label": label,
            "Wind_Speed_Vw": vw,
            "Num_Samples": num_rows,
            "Columns": ['Time', 'Amplitude_Voltage_1'],
            "RMS": rms,
            "Kurtosis": kurt,
            "Skewness": skw,
            "Peak_to_Peak": peak_to_peak,
            "Mean": mean_val,
            "Std": std_val
        })
        
        # Vẽ biểu đồ cho tệp đầu tiên làm ví dụ
        if len(data_summary) == 1:
            plt.figure(figsize=(10, 4))
            plt.plot(df['Time'], vibration, label=f"Signal (File: {filename})")
            plt.title("Vibration Signal Example")
            plt.xlabel("Time (s)")
            plt.ylabel("Amplitude (Voltage)")
            plt.legend()
            plt.savefig('vibration_example.png')
            plt.close()
    
    return data_summary

def main():
    # Đọc và phân tích dữ liệu
    data_summary = read_and_analyze_data(DATA_DIR, FILE_NAMES)
    
    if not data_summary:
        print("Không tìm thấy dữ liệu hợp lệ trong thư mục:", DATA_DIR)
        return
    
    # Chuyển thông tin thành DataFrame
    summary_df = pd.DataFrame(data_summary)
    
    # In thông tin tổng quan
    print(f"Tổng số tệp đọc được: {len(data_summary)}")
    print(f"Nhãn tìm thấy: {summary_df['Label'].unique()}")
    print(f"Tốc độ gió (Vw) tìm thấy: {summary_df['Wind_Speed_Vw'].unique()}")
    print(f"Số mẫu trung bình mỗi tệp: {summary_df['Num_Samples'].mean():.2f}")
    print("\nThông tin đặc trưng thống kê:")
    print(summary_df[['RMS', 'Kurtosis', 'Skewness', 'Peak_to_Peak', 'Mean', 'Std']].describe())
    
    # Lưu thông tin vào tệp CSV
    summary_df.to_csv('data_summary.csv', index=False)
    print("Đã lưu thông tin phân tích vào 'data_summary.csv'")
    
    # Vẽ biểu đồ phân bố RMS theo nhãn và tốc độ gió
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

if __name__ == "__main__":
    main()