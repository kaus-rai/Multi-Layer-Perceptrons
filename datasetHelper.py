import tensorflow as tf
import numpy as np
from sklearn.preprocessing import LabelBinarizer

#Loading the MNIST dataset
def loadDataset(flatten=False):
    mnistDataset = tf.keras.datasets.mnist
    (X_train, y_train), (X_test, y_test) =  mnistDataset.load_data()

    #Performing normalization on X
    X_train = X_train.astype(float)/255
    X_test = X_train.astype(float)/255

    #Creating a validaton dataset
    X_train, X_val = X_train[:-10000], X_train[-10000:]
    y_train, y_val = y_train[:-10000], y_train[-10000:]

    if flatten:
        X_train = X_train.reshape([X_train.shape[0], -1])
        X_val = X_val.reshape([X_val.shape[0], -1])
        X_test = X_test.reshape([X_test.shape[0], -1])

    return X_train, y_train, X_val, y_val, X_test, y_test


#Function to change the dimensions of the image
def changeDimensionality(X, img_h, img_w):
    X = X.reshape((X.shape[0], img_h * img_w))

    return X

#Function to perform one-hot-encoding
def hotEncoding(labels):
    labelBinaryObj = LabelBinarizer()

    return labelBinaryObj.fit_transform(labels)
