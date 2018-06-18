import numpy as np
from sklearn import svm, model_selection, neighbors, neural_network, naive_bayes, tree, feature_selection
from sklearn.metrics import accuracy_score, confusion_matrix, jaccard_similarity_score, f1_score, recall_score, precision_score
import class_distribution as cdist

def classification_with_SVM(pxl_train, label_train, pxl_test, kernel='rbf'):
    clf = svm.SVC(kernel=kernel)
    clf.fit(pxl_train, label_train)
    predictions = clf.predict(pxl_test)
    return predictions


def classification_with_nearest_neighbors(pxl_train, label_train, pxl_test, k):
    clf = neighbors.KNeighborsClassifier(n_neighbors=k)
    clf.fit(pxl_train, label_train)
    predictions = clf.predict(pxl_test)
    return predictions


def classification_with_naive_bayes(pxl_train, label_train, pxl_test):
    clf = naive_bayes.GaussianNB()
    clf.fit(pxl_train, label_train)
    predictions = clf.predict(pxl_test)
    return predictions


def classification_with_decision_tree(pxl_train, label_train, pxl_test):
    clf = tree.DecisionTreeClassifier()
    clf.fit(pxl_train, label_train)
    predictions = clf.predict(pxl_test)
    return predictions


def classification_with_MLP(pxl_train, label_train, pxl_test, hidden_layer=(5, 2)):
    clf = neural_network.MLPClassifier(hidden_layer_sizes=hidden_layer, max_iter=500)
    clf.fit(pxl_train, label_train)
    predictions = clf.predict(pxl_test)
    return predictions


def calculate_metrics(name, predictions, label_test):
    acc = accuracy_score(y_true=label_test, y_pred=predictions)
    conf = confusion_matrix(y_true=label_test, y_pred=predictions)
    js = jaccard_similarity_score(y_true=label_test, y_pred=predictions)
    f1 = f1_score(y_true=label_test, y_pred=predictions, average='macro')
    r = recall_score(y_true=label_test, y_pred=predictions, average='macro')
    p = precision_score(y_true=label_test, y_pred=predictions, average='macro')
    result = 'Acc:', acc, 'F-measure: ', f1, 'Recall: ', r, 'Precision: ', p, 'Confusion Matrix:', \
             conf
    file = open('results/' + name + '.txt', 'w+')
    file.write(str(result))
    file.close()


if __name__ == '__main__':
    h = 500
    w = int(h * 1.014)

    cdist.cdist_main()

    #TODO Tüm pixel_x.npy'lerle çalıştır
    conv = [1, 3, 5, 9, 11, 13]

    for i in conv:
        pixels = np.load('selected_pixels_' + str(i) + '.npy')
        labels = np.load('selected_labels.npy')

        pixels = np.squeeze(pixels, 2)

        # return label index
        # 0010000000 -> 2.label
        labels = np.argmax(labels, axis=1)

        # TODO AYNI TEST VE TRAN İÇİN SONUCLARI KARŞILASTIR !! MLP BİTMİYORI
        pxl_train, pxl_test, label_train, label_test = model_selection.train_test_split(pixels, labels, test_size=0.33)

        svm_kernels = ['rbf', 'poly', 'linear', 'sigmoid']
        k_neighbors = [3, 5, 7]
        hidden_layer = [(20, 20, 20, 20), (40, 40, 40, 40)]


        for k in k_neighbors:
            predictions = classification_with_nearest_neighbors(pxl_train, label_train, pxl_test, k)
            calculate_metrics(str(i) + '_' + str(k)+'NN', predictions, label_test)

        predictions = classification_with_naive_bayes(pxl_train, label_train, pxl_test)
        calculate_metrics(str(i) + '_' + 'bayes', predictions, label_test)

        predictions = classification_with_decision_tree(pxl_train, label_train, pxl_test)
        calculate_metrics(str(i) + '_' + 'decision_tree', predictions, label_test)

        for layer, j in zip(hidden_layer, range(2)):
            predictions = classification_with_MLP(pxl_train, label_train, pxl_test, layer)
            calculate_metrics(str(i) + '_' + str(j)+'_MLP', predictions, label_test)


    for i in conv:
        pixels = np.load('selected_pixels_' + str(i) + '.npy')
        labels = np.load('selected_labels.npy')

        pixels = np.squeeze(pixels, 2)

        # return label index
        # 0010000000 -> 2.label
        labels = np.argmax(labels, axis=1)

        # TODO AYNI TEST VE TRAN İÇİN SONUCLARI KARŞILASTIR !! MLP BİTMİYORI
        pxl_train, pxl_test, label_train, label_test = model_selection.train_test_split(pixels, labels, test_size=0.33)

        svm_kernels = ['rbf', 'poly', 'linear', 'sigmoid']

        for kernel in svm_kernels:
            print(kernel + 'SVM')
            predictions = classification_with_SVM(pxl_train, label_train, pxl_test, kernel)
            calculate_metrics(str(i) + '_' + kernel + '_SVM', predictions, label_test)