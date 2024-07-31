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
selected_features_rfe = ['average_line_count', 'average_param_count', 'average_var_count']
X_rfe_log_transformed = np.log1p(data_encoded[selected_features_rfe])  # np.log1p 是 log(x + 1) 的简写

# 创建新的数据集，保留未变换的特征
dataset_RFE = data_encoded[all_features].copy()
dataset_RFE[selected_features_rfe] = X_rfe_log_transformed

# 添加目标变量
dataset_RFE['pattern'] = data_encoded['pattern']

# 保存新的数据集
dataset_RFE.to_csv('dataset_RFE.csv', index=False)
