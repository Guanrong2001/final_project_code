import pandas as pd
import numpy as np

# 读取数据
file_path = 'dataset_new.csv'
data = pd.read_csv(file_path)

# 转换目标变量 'pattern' 为数值编码
data_encoded = data.copy()
data_encoded['pattern'] = data_encoded['pattern'].astype('category').cat.codes

# 选择所有特征，排除目标变量
all_features = data_encoded.columns.drop('pattern')

# 对选择的特征进行对数变换
selected_features_rffi = ['average_line_count', 'method_count', 'total_line_count']
X_rffi_log_transformed = np.log1p(data_encoded[selected_features_rffi])  # np.log1p 是 log(x + 1) 的简写

# 创建新的数据集，保留未变换的特征
dataset_RFFI = data_encoded[all_features].copy()
dataset_RFFI[selected_features_rffi] = X_rffi_log_transformed

# 添加目标变量
dataset_RFFI['pattern'] = data_encoded['pattern']

# 保存新的数据集
dataset_RFFI.to_csv('dataset_RFFI.csv', index=False)
