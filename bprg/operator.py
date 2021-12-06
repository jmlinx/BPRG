import tensorflow as tf
import numpy as np

from .kernel import KernelMesh

DTYPE = 'float32'

class Operator:
    def __init__(self, S, ns, κ, K, μ, η):
        S_lin = tf.linspace(S[0], S[1], ns+1)
        S_lin = tf.cast(S_lin, dtype=DTYPE)[:, None] # as column vector
        self.S_lin = S_lin
        self.μ = μ
        # self.κ_plain = κ
        self.κ = KernelMesh(κ)
        
        dμ = μ(S_lin[1:]) - μ(S_lin[:-1])
        dμ = np.append(dμ, dμ[-1])[:, None] # as column vector
        self.dμ = dμ
        
        self.K = K
        self.η = tf.cast(η, dtype=DTYPE)[:, None] # as column vector

    def Λ(self, f_S, x):
        vec = f_S*self.dμ
        mat_T = tf.transpose(self.κ(self.S_lin, x))
        riemann_sum = tf.matmul(mat_T, vec)
        return riemann_sum
    
    def P(self, k, f_S, x):
        λ = self.Λ(f_S, x)
        p = λ**k / np.math.factorial(k) * tf.math.exp(-λ)
        return p
    
    def Ψ(self, f_S, x):
        P_list = tf.concat([self.P(k, f_S, x) for k in range(self.K)], axis=1)
        zeros = tf.zeros_like(P_list[:,0, None])
        P_list = tf.concat([zeros, P_list], axis=1)
        P_list_cumsum = tf.cumsum(P_list, axis=1)
        ψ = tf.matmul((1-P_list_cumsum), self.η)
        return ψ