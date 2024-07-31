import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# 加载数据集
dataset_path = 'dataset.csv'
input_1300_path = 'input-1300.csv'

dataset = pd.read_csv(dataset_path)
input_1300 = pd.read_csv(input_1300_path)

# 打印数据集的前几行以确保数据加载正确
print("Dataset.csv 前几行：")
print(dataset.head())

print("\nInput-1300.csv 前几行：")
print(input_1300.head())

# 特征和标签
X = dataset.drop(['pattern'], axis=1)
y = dataset['pattern']

# 检查字符串列
string_columns = X.select_dtypes(include=['object']).columns
print(f"\n字符串列: {string_columns}")

# 创建预处理管道
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), X.select_dtypes(exclude=['object']).columns),
        ('cat', OneHotEncoder(handle_unknown='ignore'), string_columns)
    ])

# 创建训练管道，使用神经网络分类器
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42))
])

# 分割数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练模型
pipeline.fit(X_train, y_train)

# 预测和评估
y_pred = pipeline.predict(X_test)
classification_rep = classification_report(y_test, y_pred)
confusion_mat = confusion_matrix(y_test, y_pred)

# 打印分类报告和混淆矩阵
print("\n分类报告：")
print(classification_rep)

print("\n混淆矩阵：")
print(confusion_mat)

# 保存结果到txt文件
with open('neural_network_results.txt', 'w') as f:
    f.write("分类报告：\n")
    f.write(classification_rep)
    f.write("\n混淆矩阵：\n")
    f.write(str(confusion_mat))

# 保存模型
joblib.dump(pipeline, 'neural_network_model.pkl')
print("\n模型已保存为 neural_network_model.pkl")
print("\n结果已保存为 neural_network_results.txt")