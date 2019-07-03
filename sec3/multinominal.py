"""
scikit-learnとTensorflowによる実践機械学習
p.96
3.4 多クラス分類

"""
import numpy as np

from sklearn.datasets.base import get_data_home 
print(get_data_home())

from sklearn.datasets import fetch_openml
X, y = fetch_openml('mnist_784', return_X_y=True)

#yを整数に変換 ex. '3' -> 3
y = y.astype(np.int)
print(X.shape, y.shape)



#データ分割
train_ind = 10000
X_train, X_test, y_train, y_test = X[:train_ind], X[train_ind:], y[:train_ind], y[train_ind:]


#データの順番をシャッフル
np.random.seed(42) #seed固定により、結果を固定
shuffle_index = np.random.permutation(train_ind) 
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]


some_digit, digit_number = X_train[0], y_train[0]
print('some digit is {}'.format(digit_number))

#SGD
from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(random_state=42) #random_state:SGDのランダムシード固定

'''
sgd_clf.fit(X_train, y_train) #y_train5ではない　=>10個のクラスがある（水面下で10個の二項分類器器を訓練している）
some_digit_scores = sgd_clf.decision_function([some_digit])
print('score:', some_digit_scores)

print('cls:', np.argmax(some_digit_scores), sgd_clf.classes_)

#cross val
from sklearn.model_selection import cross_val_score
#val = cross_val_score(sgd_clf, X_train, y_train, cv=3, scoring='accuracy')
#print('cross valid score:', val) #[0.85521583 0.86956522 0.86362271]
'''

#入力値のスケーリング
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_Scaled = scaler.fit_transform(X_train.astype(np.float64))
#val = cross_val_score(sgd_clf, X_train_Scaled, y_train, cv=3, scoring='accuracy')
#print('cross valid score (X scaled):', val) #[0.90197842 0.90734633 0.89996996]


#3.5 誤分類の分析
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix

y_train_pred = cross_val_predict(sgd_clf, X_train_Scaled, y_train, cv=3)
conf_mx = confusion_matrix(y_train, y_train_pred)

#割合に変換
conf_mx = conf_mx / conf_mx.sum(axis=1, keepdims=True)
print('混合行列 : ¥n ', conf_mx)

