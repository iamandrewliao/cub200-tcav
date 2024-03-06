from pathlib import Path
from typing import Union

import tensorflow as tf
import numpy as np
from sklearn.linear_model import SGDClassifier


class TCAV(object):
    """tcav
    init with model
    then use_bottleneck
    then train_cav
    then calculate_sensitivty
    """

    def __init__(self, model, lm=None):
        self.model = model

        self.lm = SGDClassifier() if not lm else lm

    def use_bottleneck(self, bottleneck: int):
        """split the model into pre and post models for tcav linear model

        Args:
            layer (int): layer to split nn model
        """

        if bottleneck < 0 or bottleneck >= len(self.model.layers):
            raise ValueError("Bottleneck layer must be greater than or equal to 0 and less than the number of layers!")

        self.model_f = tf.keras.Model(inputs=self.model.input, outputs=self.model.layers[bottleneck].output)

        # create model h functional
        model_h_input = tf.keras.layers.Input(self.model.layers[bottleneck + 1].input_shape[1:])
        model_h = model_h_input
        for layer in self.model.layers[bottleneck + 1 :]:
            model_h = layer(model_h)
        self.model_h = tf.keras.Model(inputs=model_h_input, outputs=model_h)
        self.bottleneck_layer = self.model.layers[bottleneck]

    def train_cav(self, concepts, counterexamples):

        concept_activations = self.model_f.predict(concepts)
        counterexamples_activations = self.model_f.predict(counterexamples)

        x = np.concatenate([concept_activations, counterexamples_activations])
        x = x.reshape(x.shape[0], -1)

        y = np.concatenate([np.ones(len(concept_activations)), np.zeros(len(counterexamples_activations))])

        self.lm.fit(x, y)
        self.coefs = self.lm.coef_
        self.cav = np.transpose(-1 * self.coefs)

    def calculate_sensitivty(self, concepts, counterexamples):

        # if we only pass in one and the model is set up for batches
        if len(concepts.shape) != len(self.model_f.input_shape):
            concepts = concepts.reshape(-1, *concepts.shape)

        if len(counterexamples.shape) != len(self.model_f.input_shape):
            counterexamples = counterexamples.reshape(-1, *counterexamples.shape)

        concept_activations = self.model_f.predict(concepts)
        counterexamples_activations = self.model_f.predict(counterexamples)

        activations = np.concatenate([concept_activations, counterexamples_activations])
        labels = np.concatenate([np.ones(len(concept_activations)), np.zeros(len(counterexamples_activations))])
        labels = labels.reshape(-1, 1)

        x_tensor = tf.convert_to_tensor(activations, dtype=tf.float32)
        y_tensor = tf.convert_to_tensor(labels, dtype=tf.float32)

        with tf.GradientTape() as tape:
            tape.watch(x_tensor)

            y_out = self.model_h(x_tensor)
            loss = tf.keras.backend.binary_crossentropy(y_tensor, y_out)

        grad = tape.gradient(loss, x_tensor)

        grad_vals = tf.keras.backend.get_value(grad)

        # reshape since cav is a vector
        self.sensitivity = np.dot(grad_vals.reshape(grad_vals.shape[0], -1), self.cav)
        self.labels = labels
        self.grad_vals = grad_vals

    def _calculate_sensitivty(self, x, y):
        activations = self.model_f.predict(x)
        labels = y.reshape(-1, 1)

        x_tensor = tf.convert_to_tensor(activations, dtype=tf.float32)
        y_tensor = tf.convert_to_tensor(labels, dtype=tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(x_tensor)
            y_out = self.model_h(x_tensor)
            loss = tf.keras.backend.binary_crossentropy(y_tensor, y_out)

        grad = tape.gradient(loss, x_tensor)
        grad_vals = tf.keras.backend.get_value(grad)
        sensitivity = np.dot(grad_vals.reshape(grad_vals.shape[0], -1), self.cav)
        return sensitivity

    def calculate_sensitivty_old(self, x_train, y_train):
        #     # i have no idea what is the way to calculate this.
        # I assumed it was just some metric but based on
        # https://github.com/pnxenopoulos/cav-keras/blob/master/cav/tcav.py
        # they are doing some thing more complicated

        tf.compat.v1.disable_eager_execution()

        if len(x_train.shape) != len(self.model_f.input_shape):
            # if we only pass in one and the model is set up for batches
            x_train = x_train.reshape(-1, *x_train.shape)

        model_f_activations = self.model_f.predict(x_train)
        reshaped_labels = y_train.reshape(-1, 1)
        tf_y_labels = tf.convert_to_tensor(reshaped_labels, dtype=np.float32)
        loss = tf.keras.backend.binary_crossentropy(tf_y_labels, self.model_h.output)
        grad = tf.keras.backend.gradients(loss, self.model_h.input)
        gradient_func = tf.keras.backend.function([self.model_h.input], grad)
        calc_grad = gradient_func([model_f_activations])[0]

        calc_grad = calc_grad.reshape(calc_grad.shape[0], -1)
        self.sensitivity_old = np.dot(calc_grad, self.cav)

    def print_sensitivity_old(self):
        # for checking.
        # from:
        # https://github.com/pnxenopoulos/cav-keras/blob/master/cav/tcav.py
        print(
            "The sensitivity of class 1 is ",
            str(np.sum(self.sensitivity[np.where(self.labels == 1)[0]] > 0) / np.where(self.labels == 1)[0].shape[0]),
        )
        print(
            "The sensitivity of class 0 is ",
            str(np.sum(self.sensitivity[np.where(self.labels == 0)[0]] > 0) / np.where(self.labels == 0)[0].shape[0]),
        )

    def sensitivity_score(self):
        """Print the sensitivities in a readable way"""

        val_label_0 = len(self.sensitivity[(self.labels == 0) & (self.sensitivity > 0)]) / len(self.labels[self.labels == 0])
        val_label_1 = len(self.sensitivity[(self.labels == 1) & (self.sensitivity > 0)]) / len(self.labels[self.labels == 1])

        # print(f"sen-class 0 to concept: {val_label_0}, sen-class 1 to concept: {val_label_1}")

        return val_label_0, val_label_1
