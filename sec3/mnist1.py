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

np.random.seed(42) #seed固定により、結果を固定
shuffle_index = np.random.permutation(train_ind) 
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]

#確率的勾配法による二項分類機の訓練
#5 vs other number

y_train_5 = (y_train == 5)
#print('y_train_5', np.unique(y_train_5)) #yに含まれている値は

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
'''
#p.87
print(res)
[[53124  1455] #5以外の画像 
 [  949  4472]] #5の画像
[[TN, FP]
[FN, TP]]
'''

#Precision and Recall Rate (p.88)
#F-value 適合率と再現率の調和平均(harmonic mean)
from sklearn.metrics import precision_score, recall_score, f1_score
precision_rate = precision_score(y_train_5, y_train_pred)
recall_rate = recall_score(y_train_5, y_train_pred)
f_value = f1_score(y_train_5, y_train_pred)

# pracision: 0.754513244474439  recall: 0.8249400479616307  f-value: 0.7881565033486078
print('pracision:', precision_rate, ' recall:', recall_rate, ' f-value:', f_value) 

#Threshold閾値を決定して、プロジェクトに合った適合率　または、再現率にする。（適合率、再現率はトレードオフ）
#例えば、適合率をあげたい場合、

#y_score = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3, method="decision_function") #決定閾値

'''
y_score_Threshold = 10000
y_train_pred_90 = (y_score > y_score_Threshold)
print(y_score, max(y_score), min(y_score))
precision_rate = precision_score(y_train_5, y_train_pred_90)
recall_rate = recall_score(y_train_5, y_train_pred_90)
f_value = f1_score(y_train_5, y_train_pred_90)

# pracision: 0.9737302977232924  recall: 0.20512820512820512  f-value: 0.3388694194728021
print('pracision:', precision_rate, ' recall:', recall_rate, ' f-value:', f_value) 
'''

#ROC曲線
#AUC ROC曲線の下の面積　完璧な分類機 1 , 無作為: 0.5
'''
from sklearn.metrics import roc_curve, roc_auc_score
fpr, tpr, threshold = roc_curve(y_train_5, y_score)

roc_auc = roc_auc_score(y_train_5, y_score)
print('ROC AUC: ', roc_auc) #ROC AUC:  0.9660259463088996
'''

