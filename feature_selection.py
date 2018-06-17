from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.feature_selection import RFE
import matplotlib.pyplot as plt
import numpy as np


labels = np.load("balanced_labels.npy")
pixels = np.load("balanced_pixels.npy")

pixels = np.squeeze(pixels, 2)

svc = SVC(kernel="linear", C=1)
rfe = RFE(estimator=svc, n_features_to_select=1, step=1)
rfe.fit(pixels, labels)
ranking = rfe.ranking_.reshape(pixels.shape)

plt.matshow(ranking, cmap=plt.cm.Blues)
plt.colorbar()
plt.title("Ranking of pixels with RFE")
plt.show()

