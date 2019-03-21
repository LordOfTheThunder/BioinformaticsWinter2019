import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
import warnings
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, FunctionTransformer
import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.dummy import DummyClassifier
from sklearn.cluster import KMeans

samples=['GSM1338298', 'GSM1338302', 'GSM1338306', 'GSM1338297', 'GSM1338301', 'GSM1338305', 'GSM1338296', 'GSM1338300',
         'GSM1338304', 'GSM1338295', 'GSM1338299', 'GSM1338303']

plt.rcParams['font.size'] = 14
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['lines.linewidth'] = 2

if __name__ == "__main__":
    # warnings.filterwarnings("ignore")
    #
    df = pd.read_csv("data_labeled.csv")
    data = df[df.columns[0:13]]

    labels = df['apoptosis_related']

    # X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=.15, stratify=labels)
    #
    # X_train = X_train.assign(label = y_train)
    # X_train.to_csv('train_mean_2.csv')
    #
    # X_test = X_test.assign(label = y_test)
    # X_test.to_csv('test_mean_2.csv')

    # train = pd.read_csv("train_undersampled_05_.csv")
    # train = pd.read_csv("train_undersampled_05_.csv")
    train = pd.read_csv("train_oversampled_1.csv")
    test = pd.read_csv("test_norm.csv")
    # train = pd.read_csv("train_mean.csv")
    # test = pd.read_csv("test_mean.csv")

    # diff_1 = train[samples[0]] - train[samples[3]]
    # diff_2 = train[samples[1]] - train[samples[4]]
    # diff_3 = train[samples[2]] - train[samples[5]]
    # diff_4 = train[samples[6]] - train[samples[9]]
    # diff_5 = train[samples[7]] - train[samples[10]]
    # diff_6 = train[samples[8]] - train[samples[11]]
    #
    # diff_1_test = test[samples[0]] - test[samples[3]]
    # diff_2_test = test[samples[1]] - test[samples[4]]
    # diff_3_test = test[samples[2]] - test[samples[5]]
    # diff_4_test = test[samples[6]] - test[samples[9]]
    # diff_5_test = test[samples[7]] - test[samples[10]]
    # diff_6_test = test[samples[8]] - test[samples[11]]
    #
    # data = {'diff_1': diff_1, 'diff_2': diff_2, 'diff_3': diff_3, 'diff_4': diff_4, 'diff_5': diff_5, 'diff_6': diff_6}
    # X_train = pd.DataFrame.from_dict(data)
    #
    # data_ = {'diff_1': diff_1_test, 'diff_2': diff_2_test, 'diff_3': diff_3_test, 'diff_4': diff_4_test,
    #          'diff_5': diff_5_test, 'diff_6': diff_6_test}
    # X_test = pd.DataFrame.from_dict(data_)

   # indices = [0, 1, 3,4, 6, 7, 9, 10]
   #  X_train = train[[samples[i] for i in indices]]
    X_train = train[samples]
    # X_test = test[[samples[i] for i in indices]]
    X_test = test[samples]
    y_train = train['label']
    y_test = test['label']

    # Standardize
    # scaler = StandardScaler().fit(X_train)
    # X_train = scaler.transform(X_train)
    # X_test = scaler.transform(X_test)

    # log-scale
    # log_transform = FunctionTransformer(np.log1p)
    # X_train = log_transform.transform(X_train)
    # X_test = log_transform.transform(X_test)

    # Create decision tree
    clf = DecisionTreeClassifier(max_depth=70)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    dt_acc = accuracy_score(y_test, y_pred)
    dt_rec = recall_score(y_test, y_pred)
    dt_prec = precision_score(y_test, y_pred)

    print("Decision Tree")
    print("accuracy score: ", dt_acc)
    print("recall score: ", dt_rec)
    print("precision score: ", dt_prec)


    # Create random forest
    # rf = RandomForestClassifier()
    # rf.fit(X_train, y_train)
    # y_pred_rf = rf.predict(X_test)
    #
    # ab = AdaBoostClassifier()
    # ab.fit(X_train, y_train)
    # y_pred_ab = ab.predict(X_test)
    #
    # Classify with SVM
    clf = svm.LinearSVC(max_iter=20)
    clf.fit(X_train, y_train)
    y_pred_svm = clf.predict(X_test)

    # clf = svm.SVC()
    # clf.fit(X_train, y_train)
    # y_pred_svc = clf.predict(X_test)

    svm_acc = accuracy_score(y_test, y_pred_svm)
    svm_rec = recall_score(y_test, y_pred_svm)
    svm_prec = precision_score(y_test, y_pred_svm)

    print("SVM")
    print("accuracy score: ", svm_acc)
    print("recall score: ", svm_rec)
    print("precision score: ", svm_prec)

    # KNN
    clf = KNeighborsClassifier(n_neighbors=3)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    knn_acc = accuracy_score(y_test, y_pred)
    knn_rec = recall_score(y_test, y_pred)
    knn_prec = precision_score(y_test, y_pred)

    print("KNN")
    print("accuracy score: ", knn_acc)
    print("recall score: ", knn_rec)
    print("precision score: ", knn_prec)


    train_filtered = pd.read_csv('train_filtered.csv')
    gene_exp = train_filtered[samples]
    model = KMeans(n_clusters=2)
    model.fit_predict(gene_exp)
    test_df = pd.read_csv('test_norm.csv')
    test_gene_exp = test_df[samples]
    pred = model.predict(test[samples])

    kmeans_acc = accuracy_score(y_test, pred)
    kmeans_rec = recall_score(y_test, pred)
    kmeans_prec = precision_score(y_test, pred)

    print("KMeans")
    print("accuracy score: ", kmeans_acc)
    print("recall score: ", kmeans_rec)
    print("precision score: ", kmeans_prec)

    # Dummy classifier
    clf = DummyClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    dummy_acc = accuracy_score(y_test, y_pred)
    dummy_rec = recall_score(y_test, y_pred)
    dummy_prec = precision_score(y_test, y_pred)

    print("Dummy")
    print("accuracy score: ", dummy_acc)
    print("recall score: ", dummy_rec)
    print("precision score: ", dummy_prec)

    acc = [dt_acc, svm_acc, knn_acc, kmeans_acc, dummy_acc]
    prec = [dt_prec, svm_prec, knn_prec, kmeans_prec, dummy_prec]
    rec = [dt_rec, svm_rec, knn_rec, kmeans_rec, dummy_rec]

    metrics = pd.DataFrame(
        {'Accuracy': acc,
         'Precision': prec,
         'Recall': rec
         })

    ax = metrics.plot.bar(zorder=3)
   # metrics_2.plot.bar(ax=ax)
    plt.grid(zorder=0, alpha=0.5)
    # plt.xlabel('Classifier')
    plt.xticks(np.arange(5), ('Decision\nTree', 'SVM', 'KNN', 'KMeans', 'Dummy\nclf.'), rotation='horizontal')
   # plt.legend(bbox_to_anchor=(0.9, 1), loc=2, borderaxespad=0.)
    plt.ylabel("Score")
    plt.title("Comparison of Classifier Scores")
    plt.show()

    # Classify with KNN
    # clf = KNeighborsClassifier(n_neighbors=3)
    # clf.fit(X_train, y_train)
    # y_pred_knn = clf.predict(X_test)


    # print("Random Forest")
    # print("accuracy score: ", accuracy_score(y_test, y_pred_rf))
    # print("recall score: ", recall_score(y_test, y_pred_rf))
    # print("precision score: ", precision_score(y_test, y_pred_rf))
    #
    # print("AdaBoost")
    # print("accuracy score: ", accuracy_score(y_test, y_pred_ab))
    # print("recall score: ", recall_score(y_test, y_pred_ab))
    # print("precision score: ", precision_score(y_test, y_pred_ab))

    # print("SVC")
    # print("accuracy score: ", accuracy_score(y_test, y_pred_svc))
    # print("recall score: ", recall_score(y_test, y_pred_svc))
    # print("precision score: ", precision_score(y_test, y_pred_svc))
    #
    # print("KNN with N=3")
    # print("accuracy score: ", accuracy_score(y_test, y_pred_knn))
    # print("recall score: ", recall_score(y_test, y_pred_knn))
    # print("precision score: ", precision_score(y_test, y_pred_knn))

    # res = pd.DataFrame({'true label': y_test.data, 'pred': y_pred_svm})
    # res.to_csv('res_2.csv')