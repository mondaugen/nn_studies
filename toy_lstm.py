# A toy example to optimize an LSTM

import chainer
from chainer import Variable, Parameter
from numpy import *

gaus=random.standard_normal
unif=random.uniform

def weight_matrix_init(m,n,dtype='float32'):
    """
    Initialze weight matrix with Glorot and Bengio heuristic.
    Returns matrix with m rows and n columns.
    """
    return unif(-sqrt(6/(m+n)),sqrt(6/(m+n)),size=(m,n)).astype(dtype)

class LSTM_Binary(chainer.Link):

    def __init__(self, l_in, l_out, n_steps, drop_out=0.):
        """
        l_in is the length of each input vector. These vectors must be made of
        floats.
        l_out is the length of each output vector. These vectors must only
        contain integers that take on the values 0 or 1.
        n_steps is the number of input steps and output steps
        """
        
        super(LSTM_Binary, self).__init__()
        with self.init_scope():
            self.lstm0 = chainer.links.NStepLSTM(n_steps,l_in,l_out,drop_out)

    def _n_step_lstm(self):
        """
        Calls n_step_lstm using this Link's weights and bias vectors.
        """


    def __call__(self, A):
        # Call with all of A
        y_ = A @ self.x# + self.b
        #return (y - y_) ** 2.
        return y_
