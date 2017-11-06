from __future__ import print_function

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.utils import shuffle

import numpy as np

RANDOM_SEED = 3
np.random.seed(RANDOM_SEED)

images = np.load('X_train.npy')
labels = np.load('y_train.npy')
test_images = np.load('X_test.npy')
images, labels = shuffle(images, labels)

X_train = images
y_train = labels
X_test = test_images
n_features = X_train.shape[1]
n_samples = X_train.shape[0]

# the label to predict is the id of the person
y = labels
n_classes = 7

print("Total dataset size:")
print("n_samples: %d" % n_samples)
print("n_features: %d" % n_features)
print("n_classes: %d" % n_classes)

n_components = 100

print("Extracting the top %d eigenfaces from %d faces" % (n_components, X_train.shape[0]))
pca = PCA(n_components=n_components, svd_solver='randomized', whiten=True).fit(X_train)

eigenfaces = pca.components_.reshape((n_components, 50, 37))

print("Projecting the input data on the eigenfaces orthonormal basis")
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)


# Train a SVM classification model
print("Fitting the classifier to the training set")
parameters = {'C': [ 1e0, 5e0, 1e1, 5e1, 1e2, 5e2, 1e3, 5e3, 1e4, 5e4, 1e5, 5e5, 1e6, 5e6],
              'gamma': [ 0.1, 0.3, 0.6, 0.01, 0.03, 0.06, 0.001, 0.003, 0.006, 0.0001, 0.0003, 0.0006], }
clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), parameters)
clf = clf.fit(X_train_pca, y_train)
print("Best estimator found by grid search:")
print(clf.best_params_)

# Quantitative evaluation of the model quality on the test set
print("Predicting people's names on the test set")
y_pred = np.array(clf.predict(X_test_pca)).reshape(-1, 1)
n_pred = np.array([i for i in range(len(y_pred))]).reshape(-1, 1)

y_pred = np.hstack((n_pred, y_pred));
np.savetxt("y_pred.csv", np.array(y_pred), delimiter=",", fmt="%d,%d", header="ImageId,PredictedClass", comments="")
