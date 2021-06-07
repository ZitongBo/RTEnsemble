from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
from sklearn import linear_model
from reader import *


X, y = datasets.load_boston(return_X_y=True)
print(read('credit'))
# 创建一个pipeline对象
pipe = make_pipeline(
     StandardScaler(),
     # RandomForestClassifier(random_state=0)
     LogisticRegression(random_state=0)
 )
clf = linear_model.LinearRegression()
# 加载鸢尾花数据集并将其切分成训练集和测试集
# X, y = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.7, random_state=4)

# 训练整个pipeline
clf.fit(X_train, y_train)

# 我们现在可以像使用其他任何估算器一样使用它
print(mean_absolute_error(clf.predict(X_test), y_test))