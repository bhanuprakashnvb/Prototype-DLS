import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
from data_preprocessing import batch_elastic_transform  # Keep only if elastic transform is external


# Data Augmentation Parameters
sigma = 4
alpha = 20

# Training Parameters
learning_rate = 0.002
training_epochs = 1500
batch_size = 250
test_display_step = 100
save_step = 50

# MNIST Parameters
input_height = 28
input_width = input_height
n_input_channel = 1
input_size = input_height * input_width * n_input_channel
n_classes = 10
n_prototypes = 15
n_features = 40  # Latent space dimension

# Build Directories for Saving
model_folder = os.path.join(os.getcwd(), "prototype_feature_attention_model")
os.makedirs(model_folder, exist_ok=True)
img_folder = os.path.join(model_folder, "img")
os.makedirs(img_folder, exist_ok=True)

# Input Placeholders
X = tf.placeholder(dtype=tf.float32, shape=[None, input_size], name='X')
Y = tf.placeholder(dtype=tf.float32, shape=[None, n_classes], name='Y')
X_img = tf.reshape(X, shape=[-1, input_height, input_width, n_input_channel], name='X_img')

# Prototype and Feature Attention Variables
prototype_feature_vectors = tf.Variable(tf.random_uniform([n_prototypes, n_features], dtype=tf.float32), name='prototype_feature_vectors')
feature_attention_weights = tf.Variable(tf.random_uniform([n_features], minval=0, maxval=1, dtype=tf.float32), name='feature_attention_weights')
prototype_attention_weights = tf.Variable(tf.random_uniform([n_prototypes], minval=0, maxval=1, dtype=tf.float32), name='prototype_attention_weights')

# Encoder Network
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

# Distance Computation Between Features and Prototypes
def list_of_distances(X, Y):
    XX = tf.reduce_sum(tf.square(X), axis=1, keepdims=True)
    YY = tf.reduce_sum(tf.square(Y), axis=1, keepdims=True)
    distances = XX + tf.transpose(YY) - 2 * tf.matmul(X, tf.transpose(Y))
    return distances

prototype_distances = list_of_distances(attended_features, prototype_feature_vectors)

# Prototype Attention
normalized_prototype_attention = tf.nn.softmax(prototype_attention_weights, axis=0)
weighted_prototype_distances = tf.multiply(prototype_distances, normalized_prototype_attention)

# Classification Logits
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

# Training Loop
with tf.Session() as sess:
    sess.run(init)
    
    # Load MNIST Data
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

    for epoch in range(training_epochs):
        total_loss_epoch, total_acc_epoch = 0, 0
        for _ in range(mnist.train.num_examples // batch_size):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            augmented_batch_x = batch_elastic_transform(batch_x, sigma=sigma, alpha=alpha, height=input_height, width=input_width)

            _, loss, acc = sess.run([optimizer, total_loss, accuracy], feed_dict={X: augmented_batch_x, Y: batch_y})
            total_loss_epoch += loss
            total_acc_epoch += acc

        print(f"Epoch {epoch + 1}, Loss: {total_loss_epoch:.4f}, Accuracy: {total_acc_epoch:.4f}")
        
        # Save Model and Visualize Prototypes
        if (epoch + 1) % save_step == 0 or epoch == training_epochs - 1:
            saver.save(sess, os.path.join(model_folder, "model.ckpt"), global_step=epoch)
            
            # Visualize Prototypes
            prototypes = sess.run(prototype_feature_vectors)
            prototypes_img = sess.run(tf.layers.dense(prototypes, input_size, activation=tf.nn.sigmoid))
            prototypes_img = prototypes_img.reshape(-1, input_height, input_width)

            import matplotlib.pyplot as plt
            for i, proto in enumerate(prototypes_img):
                plt.subplot(3, 5, i + 1)
                plt.imshow(proto, cmap='gray')
                plt.axis('off')
            plt.savefig(os.path.join(img_folder, f"prototypes_epoch_{epoch + 1}.png"))
            plt.close()

    print("Training complete!")
