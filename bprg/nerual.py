"""
Nerual network approximation for function
"""

import numpy as np
import inspect
import re
import tensorflow as tf
from tensorflow import keras

LOSS_NAME = 'loss'
DTYPE = 'float32'
DEFAULT_FPARAMS = {
    'function': 'function',
}

class NeuralFunction(keras.Sequential):
    """
    A specific network structure for approximating a function.
    """

    def __init__(self, *args,
                 fparams=DEFAULT_FPARAMS,
                 **kwargs):
        super(NeuralFunction, self).__init__(*args, **kwargs)
        self._init_params(fparams)
        self.init_function()

    def init_function(self):
        """
        Override this function to sepecify extra trainable variables. See
        `NavierStokes` subclass in this file for referrence.
        """
        pass

    def eval_errors(self, *args, **kwargs):
        """Override this fucntion to specify the function errors/constrains to be learned. See examples
        from subclass and notebooks. 

        Returns:
            A list of error terms where each error term is an vector.
        """
        return []

    def _init_params(self, fparams):
        for key, value in fparams.items():
            setattr(self, key, value)
        self.error_names = get_return_names(self.eval_errors)

    def split_X(self, X, num_or_size_splits=None):
        """Split a matrix into multiple matrices or vectors.

        Args:
            X: a `Tensor` matrix.
            num_or_size_splits: `int` or `list`, see `tensorflw.split`.

        Returns:
            a `list` of `Tensor` matrices.
        """
        if not num_or_size_splits:
            num_or_size_splits = X.shape[-1]
        X_split = tf.split(X, num_or_size_splits, -1, name='X_split')
        X_split = [tf.cast(x, DTYPE) for x in X_split]

        return X_split

    def eval_function(self, *X_split, training=True):
        """A utility function to customize `tensorflow.keras.Model.call()`.
        Its main purpose is to take in the training data variable by variable
        so that the neural network's gradient with respect to each variable can
        be captured by `tensorflow.GradientTape()`.

        Args:
            *X_split (`numpy.array` or `Tensor`): Splited training data.               
            training (bool, optional): Whether or not to train the model. It
                serves the same purpose as the `training` option in
                `tensorflow.keras.Model.call()`.

        Returns:
            Predicted value by the neural network.
        """
        X = tf.concat(X_split, -1)
        return self(X, training=training)

    def eval_losses(self, X, y):
        """Evaluate losses from PDE equations.

        Args:
            X (`numpy.array` or `Tensor`): Tranining data.
            y (`numpy.array` or `Tensor`): A zero vector placeholder.

        Returns:
            A list of scalers corresponding to losses from PDE equations.
        """
        errors = self.eval_errors(X, y)
        losses = [self.compiled_loss(err, tf.zeros_like(err))
                      for err in errors]
        return losses

    @tf.function
    def train_step(self, data):
        """This function customizes the `tensorflow.keral.Model.fit()` behavior
        to facilitate learning PDE.

        Args:
            data (`list`): (X, y)

        Returns:
            `dict`:  a log dictionary. See `Tensorflow - Customize what happens
            in Model.fit`
            (https://www.tensorflow.org/guide/keras/customizing_what_happens_in_fit).
        """
        X, y = data
        y_pred = self(X, training=True)

        # calculate losses
        with tf.GradientTape() as tape:
            tape.watch(X)
            losses = self.eval_losses(X, y)
            total_loss = tf.reduce_sum(losses)

        # Compute gradients
        grads = tape.gradient(total_loss, self.trainable_variables)

        # Update weights
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        # Return a dict mapping metric names to current value
        self.compiled_metrics.update_state(y, y_pred)

        # make log
        log = {}
        log.update({'%s_%s' % (LOSS_NAME, name): loss
                    for name, loss in zip(self.error_names, losses)})

        log.update({'%s_total' % LOSS_NAME: total_loss})

        return log

    @tf.function
    def test_step(self, data):
        """This function customizes the `tensorflow.keral.Model.fit()` behavior
        to facilitate learning PDE.

        Args:
            data (`list`): (X, y)

        Returns:
            `dict`:  a log dictionary. See `Tensorflow - Customize what happens
            in Model.fit`
            (https://www.tensorflow.org/guide/keras/customizing_what_happens_in_fit).
        """
        X, y = data
        y_pred = self(X, training=False)

        losses = self.eval_losses(X, y)
        total_loss = tf.reduce_sum(losses)

        # Return a dict mapping metric names to current value
        self.compiled_metrics.update_state(y, y_pred)

        # make log
        log = {}
        log.update({'%s_%s' % (LOSS_NAME, name): loss
                    for name, loss in zip(self.error_names, losses)})
        log.update({'%s_total' % LOSS_NAME: total_loss})
        return log


def get_return_names(function):
    """Get the names of returned variables from the function.

    Args:
        function: a function.

    Returns:
        a list of str of variable names of function return.
    """
    str_fn = inspect.getsource(function)
    str_fn_return = str_fn.split('return')[-1]
    to_drop = '|'.join([' ', r'\[', r'\]', '\n', '_err', '_error', '_val'])
    str_fn_return_names = re.sub(to_drop, '', str_fn_return).split(',')

    return str_fn_return_names


class FixPoint(NeuralFunction):   
    def eval_errors(self, X, y):
        op = self.__dict__['operators']
        S = op.S_lin
        
        f_S = self(S) # vector of f on S
        f_X = self(X) # vector of f on x's        
        ψf_X = op.Ψ(f_S, X)
        
        err = ψf_X - f_X
        return [err]