import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# 加载数据集
file_path = 'dataset_RFE.csv'
data = pd.read_csv(file_path)

# 准备数据
X = data.drop(columns=['pattern'])
y = data['pattern']

# 处理分类数据
X = pd.get_dummies(X, drop_first=True)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 训练随机森林分类器
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# 进行预测
y_pred = clf.predict(X_test)

# 评估模型
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# 计算混淆矩阵
conf_matrix = confusion_matrix(y_test, y_pred)

# 准备分类报告和混淆矩阵字符串
report_str = f"Accuracy: {accuracy}\n\nClassification Report:\n{report}\n\nConfusion Matrix:\n{conf_matrix}"

# 定义输出文件路径
output_file_path = "RF_RFE.txt"

# 将报告写入txt文件
with open(output_file_path, "w") as file:
    file.write(report_str)

print(f"分类报告和混淆矩阵已保存到 {output_file_path}")