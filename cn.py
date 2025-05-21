#!/usr/bin/env python
# coding: utf-8

import numpy as np
from tqdm import tqdm
import tensorflow as tf
from datetime import datetime

# Parameters based on the paper
epsilon = 1e-7
m_plus = 0.9
m_minus = 0.1
lambda_ = 0.5
alpha = 0.0005
epochs = 50
no_of_secondary_capsules = 10

optimizer = tf.keras.optimizers.Adam()

params = {
    "no_of_conv_kernels": 256,
    "no_of_primary_capsules": 32,
    "no_of_secondary_capsules": 10,
    "primary_capsule_vector": 8,
    "secondary_capsule_vector": 16,
    "r": 3,
}

checkpoint_path = './logs/model/capsule'

stamp = datetime.now().strftime("%Y%m%d-%H%M%S")

logdir = './logs/func/%s' % stamp
writer = tf.summary.create_file_writer(logdir)

scalar_logdir = './logs/scalars/%s' % stamp
file_writer = tf.summary.create_file_writer(scalar_logdir + "/metrics")

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

X_train = X_train / 255.0
X_train = tf.cast(X_train, dtype=tf.float32)
X_train = tf.expand_dims(X_train, axis=-1)

X_test = X_test / 255.0
X_test = tf.cast(X_test, dtype=tf.float32)
X_test = tf.expand_dims(X_test, axis=-1)

testing_dataset_size = X_test.shape[0]
training_dataset_size = X_train.shape[0]

dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
dataset = dataset.shuffle(buffer_size=len(dataset), reshuffle_each_iteration=True)
dataset = dataset.batch(batch_size=64)

testing = tf.data.Dataset.from_tensor_slices((X_test, y_test))
testing = testing.batch(batch_size=64)

class CapsuleNetwork(tf.keras.Model):
    def __init__(self, no_of_conv_kernels, no_of_primary_capsules, primary_capsule_vector, no_of_secondary_capsules, secondary_capsule_vector, r):
        super(CapsuleNetwork, self).__init__()
        self.no_of_conv_kernels = no_of_conv_kernels
        self.no_of_primary_capsules = no_of_primary_capsules
        self.primary_capsule_vector = primary_capsule_vector
        self.no_of_secondary_capsules = no_of_secondary_capsules
        self.secondary_capsule_vector = secondary_capsule_vector
        self.r = r
        
        self.convolution = tf.keras.layers.Conv2D(self.no_of_conv_kernels, [9,9], strides=[1,1], name='ConvolutionLayer', activation='relu')
        self.primary_capsule = tf.keras.layers.Conv2D(self.no_of_primary_capsules * self.primary_capsule_vector, [9,9], strides=[2,2], name="PrimaryCapsule")
        self.w = tf.Variable(tf.random_normal_initializer()(shape=[1, 1152, self.no_of_secondary_capsules, self.secondary_capsule_vector, self.primary_capsule_vector]), dtype=tf.float32, name="PoseEstimation", trainable=True)
        self.dense_1 = tf.keras.layers.Dense(units=512, activation='relu')
        self.dense_2 = tf.keras.layers.Dense(units=1024, activation='relu')
        self.dense_3 = tf.keras.layers.Dense(units=784, activation='sigmoid', dtype='float32')
        
    def build(self, input_shape):
        pass
        
    def squash(self, s):
        s_norm = tf.norm(s, axis=-1, keepdims=True)
        return tf.square(s_norm)/(1 + tf.square(s_norm)) * s/(s_norm + epsilon)
    
    @tf.function
    def call(self, inputs):
        input_x, y = inputs
        
        x = self.convolution(input_x) # x.shape: (None, 20, 20, 256)
        x = self.primary_capsule(x) # x.shape: (None, 6, 6, 256)
        
        u = tf.reshape(x, (-1, self.no_of_primary_capsules * x.shape[1] * x.shape[2], 8)) # u.shape: (None, 1152, 8)
        u = tf.expand_dims(u, axis=-2) # u.shape: (None, 1152, 1, 8)
        u = tf.expand_dims(u, axis=-1) # u.shape: (None, 1152, 1, 8, 1)
        u_hat = tf.matmul(self.w, u) # u_hat.shape: (None, 1152, 10, 16, 1)
        u_hat = tf.squeeze(u_hat, [4]) # u_hat.shape: (None, 1152, 10, 16)

        b = tf.zeros((input_x.shape[0], 1152, self.no_of_secondary_capsules, 1)) # b.shape: (None, 1152, 10, 1)
        for i in range(self.r):
            c = tf.nn.softmax(b, axis=-2) # c.shape: (None, 1152, 10, 1)
            s = tf.reduce_sum(tf.multiply(c, u_hat), axis=1, keepdims=True) # s.shape: (None, 1, 10, 16)
            v = self.squash(s) # v.shape: (None, 1, 10, 16)
            agreement = tf.squeeze(tf.matmul(tf.expand_dims(u_hat, axis=-1), tf.expand_dims(v, axis=-1), transpose_a=True), [4]) # agreement.shape: (None, 1152, 10, 1)
            b += agreement
                
        y = tf.expand_dims(y, axis=-1) # y.shape: (None, 10, 1)
        y = tf.expand_dims(y, axis=1) # y.shape: (None, 1, 10, 1)
        mask = tf.cast(y, dtype=tf.float32) # mask.shape: (None, 1, 10, 1)
        v_masked = tf.multiply(mask, v) # v_masked.shape: (None, 1, 10, 16)
            
        v_ = tf.reshape(v_masked, [-1, self.no_of_secondary_capsules * self.secondary_capsule_vector]) # v_.shape: (None, 160)
        reconstructed_image = self.dense_1(v_) # reconstructed_image.shape: (None, 512)
        reconstructed_image = self.dense_2(reconstructed_image) # reconstructed_image.shape: (None, 1024)
        reconstructed_image = self.dense_3(reconstructed_image) # reconstructed_image.shape: (None, 784)
        
        return v, reconstructed_image

    @tf.function
    def predict_capsule_output(self, inputs):
        x = self.convolution(inputs) # x.shape: (None, 20, 20, 256)
        x = self.primary_capsule(x) # x.shape: (None, 6, 6, 256)
        
        u = tf.reshape(x, (-1, self.no_of_primary_capsules * x.shape[1] * x.shape[2], 8)) # u.shape: (None, 1152, 8)
        u = tf.expand_dims(u, axis=-2) # u.shape: (None, 1152, 1, 8)
        u = tf.expand_dims(u, axis=-1) # u.shape: (None, 1152, 1, 8, 1)
        u_hat = tf.matmul(self.w, u) # u_hat.shape: (None, 1152, 10, 16, 1)
        u_hat = tf.squeeze(u_hat, [4]) # u_hat.shape: (None, 1152, 10, 16)

        b = tf.zeros((inputs.shape[0], 1152, self.no_of_secondary_capsules, 1)) # b.shape: (None, 1152, 10, 1)
        for i in range(self.r):
            c = tf.nn.softmax(b, axis=-2) # c.shape: (None, 1152, 10, 1)
            s = tf.reduce_sum(tf.multiply(c, u_hat), axis=1, keepdims=True) # s.shape: (None, 1, 10, 16)
            v = self.squash(s) # v.shape: (None, 1, 10, 16)
            agreement = tf.squeeze(tf.matmul(tf.expand_dims(u_hat, axis=-1), tf.expand_dims(v, axis=-1), transpose_a=True), [4]) # agreement.shape: (None, 1152, 10, 1)
            b += agreement
        return v

    @tf.function
    def regenerate_image(self, inputs):
        v_ = tf.reshape(inputs, [-1, self.no_of_secondary_capsules * self.secondary_capsule_vector]) # v_.shape: (None, 160)
        reconstructed_image = self.dense_1(v_) # reconstructed_image.shape: (None, 512)
        reconstructed_image = self.dense_2(reconstructed_image) # reconstructed_image.shape: (None, 1024)
        reconstructed_image = self.dense_3(reconstructed_image) # reconstructed_image.shape: (None, 784)
        
        return reconstructed_image

capsule = CapsuleNetwork(**params)

# Loss function
def margin_loss(v, y_true, y_pred):
    v = tf.squeeze(v, [1])
    max_l = tf.square(tf.maximum(0., m_plus - tf.norm(v, axis=-1)))
    max_r = tf.square(tf.maximum(0., tf.norm(v, axis=-1) - m_minus))
    max_l = tf.reshape(max_l, shape=(-1, no_of_secondary_capsules))
    max_r = tf.reshape(max_r, shape=(-1, no_of_secondary_capsules))
    T_c = y_pred
    L_c = T_c * max_l + lambda_ * (1-T_c) * max_r
    margin_loss = tf.reduce_mean(tf.reduce_sum(L_c, axis=-1))
    
    return margin_loss

# Training step
@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        v, reconstructed_image = capsule((x, y))
        y_pred = tf.one_hot(y, depth=10)
        loss = margin_loss(v, y, y_pred)
        loss += tf.reduce_mean(tf.square(x - reconstructed_image))
        
    gradients = tape.gradient(loss, capsule.trainable_variables)
    optimizer.apply_gradients(zip(gradients, capsule.trainable_variables))
    
    return loss

# Testing step
@tf.function
def test_step(x, y):
    v = capsule.predict_capsule_output(x)
    v_norm = tf.norm(v, axis=-1)
    predicted_class = tf.argmax(v_norm, axis=-1, output_type=tf.int32)
    return tf.reduce_sum(tf.cast(predicted_class == y, tf.float32))/x.shape[0]

# Training loop
for epoch in range(epochs):
    print(f'Epoch {epoch + 1}/{epochs}')
    loss = 0
    for x, y in tqdm(dataset):
        loss = train_step(x, y)
    
    test_accuracy = 0
    for x, y in testing:
        test_accuracy += test_step(x, y)
    
    test_accuracy /= len(testing)
    print(f'Loss: {loss.numpy():.4f}, Test Accuracy: {test_accuracy.numpy():.4f}')

