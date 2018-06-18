from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.feature_selection import RFE
import matplotlib.pyplot as plt
import numpy as np


labels = np.load("selected_labels.npy")
pixels = np.load("selected_pixels.npy")

pixels = np.squeeze(pixels, 2)
labels = np.argmax(labels, axis=1)

svc = SVC(kernel="linear", C=1)
rfe = RFE(estimator=svc, n_features_to_select=1, step=1)
rfe.fit(pixels, labels)
ranking = rfe.ranking_.reshape(pixels.shape)
print(ranking)


