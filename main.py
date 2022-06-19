from datasetHelper import loadDataset, changeDimensionality, hotEncoding

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