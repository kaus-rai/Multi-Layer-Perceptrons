from datasetHelper import loadDataset, changeDimensionality, hotEncoding
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score, accuracy_score

sess = tf.InteractiveSession()

#Image height and width
img_h, img_w = 28, 28

#Loading and formatting the dataset
X_train, y_train, X_val, y_val, X_test, y_test = loadDataset()

#Changing the dimensions of the image
X_train = changeDimensionality(X_train, img_h, img_w)
X_test = changeDimensionality(X_test, img_h, img_w)

print("Training Dataset Dimensions", X_train.shape)
print("Test Dataset Dimensions", X_test.shape)

#Performing hot encoding on y-labels
y_train = hotEncoding(y_train)
y_test = hotEncoding(y_test)

print("Train Label Dimensions", y_train.shape)
print("Test Label Dimensions", y_test.shape)

#Defining Parameters
classes = y_train.shape[1]
features = X_train.shape[1]
output = y_train.shape[1]
layer_0 = 512
layer_1 = 256
learning_rate = 0.001
regularizer_rate = 0.1

#Defining Placeholders