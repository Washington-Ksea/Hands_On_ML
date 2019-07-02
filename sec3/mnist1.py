"""
scikit-learnとTensorflowによる実践機械学習
p.82

"""
import numpy as np

from sklearn.datasets.base import get_data_home 
print(get_data_home())

from sklearn.datasets import fetch_openml
X, y = fetch_openml('mnist_784', return_X_y=True)

#yを整数に変換 ex. '3' -> 3
y = y.astype(np.int)
print(X.shape, y.shape)

#-----------------------------------
"""
import matplotlib.pyplot as plt
import matplotlib
some_digit = X[36000]
some_digit_img = some_digit.reshape(28, 28)
plt.imshow(some_digit_img, cmap=matplotlib.cm.binary, 
            interpolation='nearest')
plt.axis('off')
plt.show()
"""

#データ分割
train_ind = 60000
X_train, X_test, y_train, y_test = X[:train_ind], X[train_ind:], y[:train_ind], y[train_ind:]

#データの順番をシャッフル


shuffle_index = np.random.permutation(train_ind)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]

#確率的勾配法による二項分類機の訓練
#5 vs other number

y_train_5 = (y_train == 5)
print('y_train_5', np.unique(y_train_5)) #yに含まれている値は

y_test_5 = (y_test == 5)
"""
>>> import numpy as np
>>> a = np.array([1,2,3,3,4])
>>> (a==3)
array([False, False,  True,  True, False])
"""

from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(random_state=42) #random_state:SGDのランダムシード固定
#sgd_clf.fit(X_train, y_train_5)

#K分割交差検証
from sklearn.model_selection import cross_val_score
#res = cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy")
#res : [0.9617  0.9505  0.96945] 正解率により評価しているが、もともと5でないと予測していれば、約90%は正解である

#混同行列
from sklearn.model_selection import cross_val_predict
y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3) #K分割交差検証の予測結果を返す

from sklearn.metrics import confusion_matrix
res = confusion_matrix(y_train_5, y_train_pred)
print(res)
