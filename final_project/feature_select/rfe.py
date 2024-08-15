import pandas as pd
import numpy as np
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

# 读取数据
file_path = 'dataset_new.csv'
data = pd.read_csv(file_path)

# 转换目标变量 'pattern' 为数值编码
data_encoded = data.copy()
data_encoded['pattern'] = data_encoded['pattern'].astype('category').cat.codes

# 准备特征和目标变量
X = data_encoded.drop(['pattern'], axis=1).select_dtypes(include=[np.number])  # 选择数值类型的特征
y = data_encoded['pattern']

# 使用Logistic Regression模型进行RFE
model = LogisticRegression(max_iter=1000)
rfe = RFE(model, n_features_to_select=3)
fit = rfe.fit(X, y)

# 获取RFE选择的特征
selected_features = X.columns[fit.support_]

# 输出选择的特征
print("Selected Features:", selected_features.tolist())