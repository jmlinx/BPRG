import tensorflow as tf
import numpy as np

default_graph = dict(
    ns = 1000,
    dx = 1/1000,
    S = [0,1],
    K = 2,# thresholds
    η = np.array([0.1, 0, 0.9]), # dv_k / dμ
    μ = lambda x: x, # Lebsgue measure
    κ = lambda x,y: (10*tf.sqrt(x**2 + y**2) 
                    / (1+ tf.sqrt(tf.abs(x-y)))),
)

default_percolation_plot = dict(
    L = 100,
    nv = 200,
    seed = 4569
)