import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# 读取数据集
file_path = '../dataset.csv'
data = pd.read_csv(file_path)

# 数据预处理
# 删除存在缺失值的行
data_cleaned = data.dropna()

# 选择特征和标签
X = data_cleaned.drop(columns=['pattern'])
y = data_cleaned['pattern']

# 将分类变量转为数值
X = pd.get_dummies(X)

# 标准化特征
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 分割数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 训练决策树分类器
dt_classifier = DecisionTreeClassifier(random_state=42)
dt_classifier.fit(X_train, y_train)

# 预测
y_pred = dt_classifier.predict(X_test)

# 生成分类报告和混淆矩阵
classification_rep = classification_report(y_test, y_pred)
confusion_mat = confusion_matrix(y_test, y_pred)

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