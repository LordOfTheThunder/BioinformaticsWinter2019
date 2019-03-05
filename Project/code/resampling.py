import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score
from math import ceil
import matplotlib.pyplot as plt

samples=['GSM1338298', 'GSM1338302', 'GSM1338306', 'GSM1338297', 'GSM1338301', 'GSM1338305', 'GSM1338296', 'GSM1338300',
         'GSM1338304', 'GSM1338295', 'GSM1338299', 'GSM1338303']

def undersample(df, coef):
    resample_indices = df[df['label'] == False].index # examples we need to make less of
    rest_indices = df[df['label'] == True].index
    n_u = ceil(len(resample_indices) * coef) # new number of 'False' = coef * number of original 'False' indices
    resampled_indices = np.random.choice(resample_indices, n_u, replace=False)
    final_indices = np.concatenate([rest_indices, resampled_indices])
    return df.iloc[final_indices]

def oversample(df, coef):
    resample_indices = df[df['label'] == True].index # examples we need to make less of
    rest_indices = df[df['label'] == False].index
    n_u = ceil(len(rest_indices) * coef) # coef indicates how much 'True' we want in relation to 'False'
    resampled_indices = np.random.choice(resample_indices, n_u, replace=True)
    final_indices = np.concatenate([rest_indices, resampled_indices])
    return df.iloc[final_indices]


# train = pd.read_csv("train_filtered.csv")
# test = pd.read_csv("test_norm.csv")
# TODO: uncomment the above lines instead of the ones below
train = pd.read_csv("train_mean.csv")
test = pd.read_csv("test_mean.csv")

def runTest(samples):
    X_test = test[samples]
    y_test = test['label']
    acc_results, recall_results, prec_results = [], [], []
    x_values = [0.5, 0.6, 0.7, 0.8, 0.9, 1]
    for val in x_values:
        train_tmp = oversample(train, val)
        print("coef is: ", val)
        print("after oversampling, train size is: ", len(train))
        print("TRUE in train size: ", len(train[train['label'] == True]))

        X_train = train_tmp[samples]
        y_train = train_tmp['label']

        # Create decision tree
        clf = DecisionTreeClassifier()
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        acc_score = accuracy_score(y_test, y_pred)
        rec_score = recall_score(y_test, y_pred)
        prec_score = precision_score(y_test, y_pred)

        print("Decision Tree")
        print("accuracy score: ", acc_score)
        print("recall score: ", rec_score)
        print("precision score: ", prec_score)

        acc_results.append(acc_score)
        recall_results.append(rec_score)
        prec_results.append(prec_score)

    plt.title('Score as a function of oversampling coefficient')
    plt.grid(True)
    plt.xlabel('variance filter')
    plt.ylabel('score')
    plt.plot(x_values, acc_results, 'r', label="accuracy")
    plt.plot(x_values, recall_results, 'b', label="recall")
    plt.plot(x_values, prec_results, 'g', label="precision")
    plt.legend()
    plt.show()


    acc_results, recall_results, prec_results = [], [], []
    x_values = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    for val in x_values:
        train_tmp = undersample(train, val)
        print("coef is: ", val)
        print("after undersampling, train size is: ", len(train))
        print("TRUE in train size: ", len(train[train['label'] == True]))

        X_train = train_tmp[samples]
        y_train = train_tmp['label']

        # Create decision tree
        clf = DecisionTreeClassifier()
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        acc_score = accuracy_score(y_test, y_pred)
        rec_score = recall_score(y_test, y_pred)
        prec_score = precision_score(y_test, y_pred)

        print("Decision Tree")
        print("accuracy score: ", acc_score)
        print("recall score: ", rec_score)
        print("precision score: ", prec_score)

        acc_results.append(acc_score)
        recall_results.append(rec_score)
        prec_results.append(prec_score)

    plt.title('Score as a function of undersampling coefficient')
    plt.grid(True)
    plt.xlabel('variance filter')
    plt.ylabel('score')
    plt.plot(x_values, acc_results, 'r', label="accuracy")
    plt.plot(x_values, recall_results, 'b', label="recall")
    plt.plot(x_values, prec_results, 'g', label="precision")
    plt.legend()
    plt.show()

runTest(samples)
# Partial testing only on BT-549 cells
runTest(samples[0:6])
# Partial testing only on MDA-mB cells
runTest(samples[7:12])

# TODO: save the best configuration to file
# best_train.to_csv('train_resampled.csv')