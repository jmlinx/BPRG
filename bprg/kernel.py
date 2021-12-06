import tensorflow as tf

DTYPE = 'float32'

class KernelMesh():
    def __init__(self, kernel):
        '''
        kernel funciton must be calculated by `tf` not `np`.
        '''
        self.kernel = kernel
    
    def __call__(self, x, y):
        XX, YY = tf.meshgrid(x, y, indexing='ij')
        kernel_mesh = self.kernel(XX, YY)
        kernel_mesh = tf.cast(kernel_mesh, dtype=DTYPE)
        return kernel_mesh
