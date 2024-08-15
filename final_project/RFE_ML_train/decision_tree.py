import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# 读取并预处理数据集
file_path = '../dataset_RFE.csv'
new_data = pd.read_csv(file_path)

# 选择特征和标签
X_new = new_data.drop(columns=['pattern'])
y_new = new_data['pattern']

# 将分类变量转为数值
X_new = pd.get_dummies(X_new)

# 标准化特征
scaler = StandardScaler()
X_new_scaled = scaler.fit_transform(X_new)

# 分割数据集为训练集和测试集
X_new_train, X_new_test, y_new_train, y_new_test = train_test_split(X_new_scaled, y_new, test_size=0.2, random_state=42)

# 训练决策树分类器
dt_classifier_new = DecisionTreeClassifier(random_state=42)
dt_classifier_new.fit(X_new_train, y_new_train)

# 预测
y_new_pred = dt_classifier_new.predict(X_new_test)

# 生成分类报告和混淆矩阵
classification_rep = classification_report(y_new_test, y_new_pred)
confusion_mat = confusion_matrix(y_new_test, y_new_pred)

# 打印分类报告和混淆矩阵
print("\n分类报告：")
print(classification_rep)

print("\n混淆矩阵：")
print(confusion_mat)

# 保存结果到txt文件
results_path = 'decision_tree_results.txt'
with open(results_path, 'w') as f:
    f.write("分类报告：\n")
    f.write(classification_rep)
    f.write("\n混淆矩阵：\n")
    f.write(str(confusion_mat))

print("\n结果已保存为 decision_tree_results.txt")