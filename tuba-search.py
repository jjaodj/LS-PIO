import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from scipy.stats import mode


train_df = pd.read_csv('train_normalized.csv')
test_df = pd.read_csv('test_normalized.csv')


train_df['label'] = train_df['class'].apply(lambda x: 0 if x == 'normal' else 1)
test_df['label'] = test_df['class'].apply(lambda x: 0 if x == 'normal' else 1)


features_iforest = [2, 3, 4, 5, 6, 7, 12, 14, 18, 20, 22, 23, 24, 25, 27, 29, 34, 35, 36, 37, 40]
features_ocsvm = [1, 2, 4, 6, 7, 9, 11, 12, 15, 17, 19, 22, 23, 25, 27, 31, 36, 38, 39, 40]
features_lof = [2, 4, 8, 9, 10, 14, 16, 17, 18, 22, 27, 31, 39]


scaler = StandardScaler()
X_train = scaler.fit_transform(train_df.drop(['class', 'label'], axis=1))
X_test = scaler.transform(test_df.drop(['class', 'label'], axis=1))

y_train = train_df['label']
y_test = test_df['label']


iforest = IsolationForest()
iforest.fit(X_train[:, features_iforest])
pred_iforest = iforest.predict(X_test[:, features_iforest])
pred_iforest = np.where(pred_iforest == 1, 0, 1)  


ocsvm = OneClassSVM()
ocsvm.fit(X_train[:, features_ocsvm])
pred_ocsvm = ocsvm.predict(X_test[:, features_ocsvm])
pred_ocsvm = np.where(pred_ocsvm == 1, 0, 1)


lof = LocalOutlierFactor(novelty=True)
lof.fit(X_train[:, features_lof])
pred_lof = lof.predict(X_test[:, features_lof])
pred_lof = np.where(pred_lof == 1, 0, 1)
weights = {'iforest':  0.3248, 'ocsvm':0.3427, 'lof': 0.3325}


ensemble_scores = (pred_iforest * weights['iforest'] + 
                   pred_ocsvm * weights['ocsvm'] + 
                   pred_lof * weights['lof'])


predictions = np.where(ensemble_scores >= 0.4, 1, 0)


accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)
