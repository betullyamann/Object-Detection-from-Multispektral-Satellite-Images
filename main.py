import numpy as np
from sklearn import model_selection, feature_selection
from resize_and_stack_images import make_stacked_images
from create_masks import create_masks
import class_distribution as cdist
from vectorize import set_feature_and_mask_vector
import classification as cls

if __name__ == '__main__':
    h = 500
    w = int(h * 1.014)

    pixels = np.load("selected_pixels.npy")
    labels = np.load("selected_labels.npy")

    #num_of_feature = 10
    #pixels_new = feature_selection.SelectKBest(feature_selection.chi2, num_of_feature).fit_transform(pixels, labels)

    # return label index
    # 0010000000 -> 2.label
    labels = np.argmax(labels, axis=1)

    # TODO AYNI TEST VE TRAIN İÇİN SONUCLARI KARŞILASTIR !! MLP BİTMİYOR
    pxl_train, pxl_test, label_train, label_test = model_selection.train_test_split(pixels_new, labels, test_size=0.33)

    svm_kernels = ["poly", "linear", "sigmoid"]
    k_neighbors = [3, 5, 7]
    hidden_layer = [(20, 20, 20, 20), (40, 40, 40, 40)]

    make_stacked_images(w, h)
    create_masks(w, h)
    set_feature_and_mask_vector()

    for svm in svm_kernels:
        predictions = cls.classification_with_SVM(svm, pxl_train, label_train, pxl_test)
        cls.calculate_metrics(svm, predictions, label_test)

    for k in k_neighbors:
        predictions = cls.classification_with_nearest_neighbors(k, pxl_train, label_train, pxl_test)
        cls.calculate_metrics(str(k)+'NN', predictions, label_test)

    predictions = cls.classification_with_naive_bayes(pxl_train, label_train, pxl_test)
    cls.calculate_metrics('bayes', predictions, label_test)

    predictions = cls.classification_with_decision_tree(pxl_train, label_train, pxl_test)
    cls.calculate_metrics('decision_tree', predictions, label_test)

    for layer in hidden_layer:
        predictions = cls.classification_with_nearest_neighbors(layer, pxl_train, label_train, pxl_test)
        cls.calculate_metrics(str(layer)+'MLP', predictions, label_test)


