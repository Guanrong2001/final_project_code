import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

# 读取数据
file_path = 'dataset_new.csv'
data = pd.read_csv(file_path)

# 转换目标变量 'pattern' 为数值编码
data_encoded = data.copy()
data_encoded['pattern'] = data_encoded['pattern'].astype('category').cat.codes

# 准备特征和目标变量
X = data_encoded.drop(['pattern'], axis=1).select_dtypes(include=[np.number])  # 选择数值类型的特征
y = data_encoded['pattern']

# 使用RandomForest模型进行特征选择
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# 使用特征重要性进行特征选择
selector = SelectFromModel(model, threshold=-np.inf, max_features=3, prefit=True)
selected_features_rf = X.columns[selector.get_support()]

# 输出选择的特征
print("Selected Features (Random Forest):", selected_features_rf.tolist())
