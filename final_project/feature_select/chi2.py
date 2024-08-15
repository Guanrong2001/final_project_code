import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, chi2

# 读取数据
file_path = 'dataset_new.csv'
data = pd.read_csv(file_path)

# 转换目标变量 'pattern' 为数值编码
data_encoded = data.copy()
data_encoded['pattern'] = data_encoded['pattern'].astype('category').cat.codes

# 准备特征和目标变量
X = data_encoded.drop(['pattern'], axis=1).select_dtypes(include=[np.number])  # 选择数值类型的特征
y = data_encoded['pattern']

# 因为卡方检验需要非负输入数据，我们将数据进行适当处理
X_chi2 = X.copy()
X_chi2[X_chi2 < 0] = 0  # 将所有负值替换为0

# 使用卡方检验进行特征选择
chi2_selector = SelectKBest(chi2, k=3)
chi2_selector.fit(X_chi2, y)
selected_features_chi2 = X.columns[chi2_selector.get_support()]

# 输出选择的特征
print("Selected Features (Chi-Square):", selected_features_chi2.tolist())
