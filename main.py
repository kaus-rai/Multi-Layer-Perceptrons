from datasetHelper import loadDataset, changeDimensionality, hotEncoding
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score, accuracy_score

sess = tf.InteractiveSession()

#Image height and width
img_h, img_w = 28, 28

batch_size = 128
epochs=14
dropout_prob = 0.6
training_accuracy = []
training_loss = []
testing_accuracy = []

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
input_X = tf.placeholder('float32', shape=(None, features), name='input_X')
input_Y = tf.placeholder('float32', shape=(None, classes), name='input_Y')
prob = tf.placeholder(tf.float32)

#Initializing the weights and bias by normal function with std=1/sqrt(number of input features)
weights_0 = tf.Variable(tf.random_normal([features, layer_0], stddev=(1/tf.sqrt(float(features)))))
bias_0 = tf.Variable(tf.random_normal([layer_0]))

weights_1 = tf.Variable(tf.random_normal([layer_0, layer_1], stddev=(1/tf.sqrt(float(layer_0)))))
bias_1 = tf.Variable(tf.random_normal([layer_1]))

weights_2 = tf.Variable(tf.random_normal([layer_1, output], stddev=(1/tf.sqrt(float(layer_1)))))
bias_2 = tf.Variable(tf.random_normal([output]))


#Defining model 
#784(Input)-512(Hidden layer 1)-256(Hidden layer 2)-10(Output) model
hidden_output_0 = tf.nn.relu(tf.matmul(input_X, weights_0) + bias_0)
hidden_output_0_0 = tf.nn.dropout(hidden_output_0, prob)

hidden_output_1 = tf.nn.relu(tf.matmul(input_X, weights_1) + bias_1)
hidden_output_1_1 = tf.nn.dropout(hidden_output_1, prob)

pred_y = tf.sigmoid(tf.matmul(hidden_output_1_1, weights_2) + bias_2)

#Defining Loss Function using Softmax cross Entropy
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = pred_y, labels=input_Y))+regularizer_rate*(tf.reduce_sum(tf.square(bias_0))+tf.reduce_sum(tf.square(bias_1)))

#Learning Rate
learning_rate = tf.train.exponential_decay(learning_rate, 0, 5, 0.85, staircase=True)

#Adam Optimizer for finding the true weight
optmizer = tf.train.AdamOptimizer(learning_rate).minimize(loss, var_list=[weights_0, weights_1,weights_2,bias_0,bias_1,bias_2])

#Metrics Definition
correct_pred = tf.equal(tf.argmax(y_train, 1), tf.argmax(pred_y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


#Training Parameters
sess.run(tf.global_variables_initializer())
for epoch in range(epochs):
    arr = np.arange(X_train.shape[0])
    np.random.shuffle(arr)

    for index in range(0, X_train.shape[0],batch_size):
        sess.run(optmizer, {input_X : X_train[arr[index:index+batch_size]], input_Y : y_train[arr[index: index+batch_size]], prob:dropout_prob})
        training_accuracy.append(sess.run(accuracy, feed_dict={
            input_X:X_train,
            input_Y:y_train,
            prob:1
        }))

        training_loss.append(sess.run(loss, {
            input_X : X_train,
            input_Y : y_train,
            prob : 1
        }))

        testing_accuracy.append(accuracy_score(y_test.argmax(1), sess.run(pred_y, {
            input_X : X_test,
            prob  :1,
        }).argmax(1)))
        print("Epoch:{0}, Train loss: {1:.2f} Train acc: {2:.3f}, Test acc:{3:.3f}".format(epoch,training_loss[epoch],training_accuracy[epoch],testing_accuracy[epoch]))
