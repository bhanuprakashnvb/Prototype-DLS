import os
import time
import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from data_preprocessing import batch_elastic_transform  # Ensure this exists
from tensorflow.examples.tutorials.mnist import input_data

# Load MNIST Data
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

# GPU Configuration
GPUID = 1
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPUID)

# Directories for Saving Results
model_folder = os.path.join(os.getcwd(), "saved_model", "mnist_feature_prototype_attention")
os.makedirs(model_folder, exist_ok=True)
img_folder = os.path.join(model_folder, "img")
os.makedirs(img_folder, exist_ok=True)

# Training Parameters
learning_rate = 0.002
training_epochs = 100
batch_size = 250
test_display_step = 10
save_step = 20

# Elastic Deformation Parameters
sigma = 4
alpha = 20

# Network Parameters
input_height = 28
input_width = input_height
n_input_channel = 1
input_size = input_height * input_width * n_input_channel
n_classes = 10
n_prototypes = 15

# Attention Variables
n_features = 40  # Dimensionality of the latent space
feature_attention_weights = tf.Variable(tf.random_uniform([n_features], minval=0, maxval=1), name='feature_attention_weights')
prototype_attention_weights = tf.Variable(tf.random_uniform([n_prototypes], minval=0, maxval=1), name='prototype_attention_weights')

# Placeholders
X = tf.placeholder(dtype=tf.float32, shape=[None, input_size], name='X')
Y = tf.placeholder(dtype=tf.float32, shape=[None, n_classes], name='Y')
X_img = tf.reshape(X, shape=[-1, input_height, input_width, n_input_channel], name='X_img')

# Encoder
def encoder(input_img):
    conv1 = tf.layers.conv2d(input_img, filters=32, kernel_size=3, strides=2, activation=tf.nn.relu, padding='same')
    conv2 = tf.layers.conv2d(conv1, filters=32, kernel_size=3, strides=2, activation=tf.nn.relu, padding='same')
    conv3 = tf.layers.conv2d(conv2, filters=32, kernel_size=3, strides=2, activation=tf.nn.relu, padding='same')
    conv4 = tf.layers.conv2d(conv3, filters=10, kernel_size=3, strides=2, activation=tf.nn.relu, padding='same')
    flattened = tf.layers.flatten(conv4)
    latent_space = tf.layers.dense(flattened, n_features, activation=None)  # Latent feature space
    return latent_space

# Latent Space with Feature Attention
feature_vectors = encoder(X_img)
normalized_feature_attention = tf.nn.softmax(feature_attention_weights, axis=0)
attended_features = tf.multiply(feature_vectors, normalized_feature_attention, name='attended_features')

# Prototypes
prototype_feature_vectors = tf.Variable(tf.random_uniform([n_prototypes, n_features], dtype=tf.float32), name='prototype_feature_vectors')

# Distance Computation
def list_of_distances(X, Y):
    XX = tf.reduce_sum(tf.square(X), axis=1, keepdims=True)
    YY = tf.reduce_sum(tf.square(Y), axis=1, keepdims=True)
    distances = XX + tf.transpose(YY) - 2 * tf.matmul(X, tf.transpose(Y))
    return distances

prototype_distances = list_of_distances(attended_features, prototype_feature_vectors)

# Prototype Attention
normalized_prototype_attention = tf.nn.softmax(prototype_attention_weights, axis=0)
weighted_prototype_distances = tf.multiply(prototype_distances, normalized_prototype_attention)

# Classification Layer
logits = tf.matmul(weighted_prototype_distances, tf.random_uniform([n_prototypes, n_classes], dtype=tf.float32))

# Loss Function (Without R1 and R2)
class_error = tf.losses.softmax_cross_entropy(onehot_labels=Y, logits=logits)
autoencoder_error = tf.losses.mean_squared_error(labels=X, predictions=tf.layers.dense(attended_features, input_size, activation=None))
total_loss = class_error + autoencoder_error

# Optimizer
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(total_loss)

# Accuracy Metric
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Model Initialization
init = tf.global_variables_initializer()
saver = tf.train.Saver()

# Training Loop with Metrics Evaluation
with tf.Session() as sess:
    sess.run(init)

    n_train_batch = mnist.train.num_examples // batch_size
    n_valid_batch = mnist.validation.num_examples // batch_size
    n_test_batch = mnist.test.num_examples // batch_size

    for epoch in range(training_epochs):
        total_loss_epoch, total_acc_epoch = 0, 0

        for _ in range(n_train_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            elastic_batch_x = batch_elastic_transform(batch_x, sigma=sigma, alpha=alpha, height=input_height, width=input_width)

            _, loss, acc = sess.run([optimizer, total_loss, accuracy], feed_dict={X: elastic_batch_x, Y: batch_y})
            total_loss_epoch += loss
            total_acc_epoch += acc

        print(f"Epoch {epoch + 1}, Loss: {total_loss_epoch:.4f}, Accuracy: {total_acc_epoch / n_train_batch:.4f}")

        # Validation Metrics
        if (epoch + 1) % test_display_step == 0:
            valid_loss, valid_acc = 0, 0
            for _ in range(n_valid_batch):
                batch_x, batch_y = mnist.validation.next_batch(batch_size)
                loss, acc = sess.run([total_loss, accuracy], feed_dict={X: batch_x, Y: batch_y})
                valid_loss += loss
                valid_acc += acc
            print(f"Validation Loss: {valid_loss / n_valid_batch:.4f}, Accuracy: {valid_acc / n_valid_batch:.4f}")

        # Save Model and Visualize Prototypes
        if (epoch + 1) % save_step == 0 or epoch == training_epochs - 1:
            saver.save(sess, os.path.join(model_folder, "model.ckpt"), global_step=epoch)

            prototypes = sess.run(prototype_feature_vectors)
            prototypes_img = sess.run(tf.layers.dense(prototypes, input_size, activation=tf.nn.sigmoid))
            prototypes_img = prototypes_img.reshape(-1, input_height, input_width)

            # Visualize Prototypes
            for i, proto in enumerate(prototypes_img):
                plt.subplot(3, 5, i + 1)
                plt.imshow(proto, cmap='gray')
                plt.axis('off')
            plt.savefig(os.path.join(img_folder, f"prototypes_epoch_{epoch + 1}.png"))
            plt.close()

    # Test Metrics
    test_loss, test_acc = 0, 0
    for _ in range(n_test_batch):
        batch_x, batch_y = mnist.test.next_batch(batch_size)
        loss, acc = sess.run([total_loss, accuracy], feed_dict={X: batch_x, Y: batch_y})
        test_loss += loss
        test_acc += acc
    print(f"Test Loss: {test_loss / n_test_batch:.4f}, Accuracy: {test_acc / n_test_batch:.4f}")

    print("Training complete!")
