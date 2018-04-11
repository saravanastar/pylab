import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cross_validation import  train_test_split
from sklearn.preprocessing import Imputer
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics

def plot_corr(df, size=11):
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(size, size))
    ax.matshow(corr)
    plt.xticks(range(len(corr.columns)), corr.columns)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.show()


df = pd.read_csv(
    '../../Documents/doc/exercise/ml/python-understanding-machine-learning/05/demos/demos/data/pima-data.csv')


del(df['skin'])
# plot_corr(df)

diabetes_map = {True:1 , False:0}
df['diabetes'] = df['diabetes'].map(diabetes_map)

num_true= len(df.loc[df['diabetes'] == 1])

num_false = len(df[df['diabetes'] == 0])
print(" positive cases " + str((num_true/len(df['diabetes']))*100))
print(" Negative cases " + str((num_false/len(df['diabetes']))*100))


X = df.loc[:, df.columns != 'diabetes'].values
y = df['diabetes'].values
split_test_size=0.30

X_train, X_test, y_train,y_test  = train_test_split(X,y, test_size=split_test_size, random_state=42)

print("{0:0.2f}% in training set ".format((len(X_train)/len(df.index))*100))

fill_imputed = Imputer(missing_values=0, strategy="mean", axis=0)
X_train = fill_imputed.fit_transform(X_train)
X_test= fill_imputed.fit_transform(X_test)

nb_model = GaussianNB()
nb_model.fit(X_train, y_train.ravel())
# print(X_test)
nb_predict_train = nb_model.predict(X_train);
# print(X_test)
# tryli = [];
# for tet in X_test:
#     tryli.append(tet[0])
#
# print("Score:" ,nb_model.score(X_test, y_test))
# plt.scatter(tryli,y_test, c=nb_predict_train);
# # plt.scatter()
# # plt.plot(type="scatter")
# plt.xlabel("Training Data")
# plt.ylabel("Predicted Data")
# plt.show()

print("Accuracy: {0:.4f}".format(metrics.accuracy_score(y_train,nb_predict_train)))

nb_predict_test = nb_model.predict(X_test)
print("Accuracy: {0:.4f}".format(metrics.accuracy_score(y_test,nb_predict_test)))


#Confustion Matrix

print("{0}".format(metrics.confusion_matrix(y_test,nb_predict_test)))

#Classification report

print(metrics.classification_report(y_test,nb_predict_test))

#RAndome Forest

print("Random Forest")
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train,y_train.ravel())

rf_predict_train = rf_model.predict(X_train)

print("Accuracy: {0:.4f}".format(metrics.accuracy_score(y_train,rf_predict_train)))


rf_predict_test = rf_model.predict(X_test)

print("Accuracy: {0:.4f}".format(metrics.accuracy_score(y_test,rf_predict_test)))

print("{0}".format(metrics.confusion_matrix(y_test, rf_predict_test)))

print(metrics.classification_report(y_test,rf_predict_test))


print("Logistic Regression")
from sklearn.linear_model import LogisticRegression

lr_model = LogisticRegression(C=0.7, random_state=42)
lr_model.fit(X_train, y_train.ravel())
lr_predict_test = lr_model.predict(X_test)

print("{0}".format(metrics.confusion_matrix(y_test, lr_predict_test)))

print(metrics.classification_report(y_test,lr_predict_test))

print("Cross validation")

from sklearn.linear_model import LogisticRegressionCV
lr_cv_model = LogisticRegressionCV(n_jobs=-1, random_state=42,Cs=3, cv=10, refit=False, class_weight="balanced")
lr_cv_model.fit(X_train, y_train.ravel())

lr_cv_predict_test = lr_cv_model.predict(X_test)

print("{0}".format(metrics.confusion_matrix(y_test, lr_cv_predict_test)))

print(metrics.classification_report(y_test,lr_cv_predict_test))
