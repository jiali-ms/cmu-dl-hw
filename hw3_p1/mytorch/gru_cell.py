import numpy as np
from activation import *


class GRU_Cell:
    """docstring for GRU_Cell"""

    def __init__(self, in_dim, hidden_dim):
        self.d = in_dim
        self.h = hidden_dim
        h = self.h
        d = self.d
        self.x_t = 0

        self.Wzh = np.random.randn(h, h)
        self.Wrh = np.random.randn(h, h)
        self.Wh = np.random.randn(h, h)

        self.Wzx = np.random.randn(h, d)
        self.Wrx = np.random.randn(h, d)
        self.Wx = np.random.randn(h, d)

        self.dWzh = np.zeros((h, h))
        self.dWrh = np.zeros((h, h))
        self.dWh = np.zeros((h, h))

        self.dWzx = np.zeros((h, d))
        self.dWrx = np.zeros((h, d))
        self.dWx = np.zeros((h, d))

        self.z_act = Sigmoid()
        self.r_act = Sigmoid()
        self.h_act = Tanh()

        # Define other variables to store forward results for backward here
        self.z = None
        self.r = None
        self.h_tilda = None
        self.h_t = None

    def init_weights(self, Wzh, Wrh, Wh, Wzx, Wrx, Wx):
        self.Wzh = Wzh
        self.Wrh = Wrh
        self.Wh = Wh
        self.Wzx = Wzx
        self.Wrx = Wrx
        self.Wx = Wx

    def __call__(self, x, h):
        return self.forward(x, h)

    def forward(self, x, h):
        # input:
        #   - x: shape(input dim),  observation at current time-step
        #   - h: shape(hidden dim), hidden-state at previous time-step
        #
        # output:
        #   - h_t: hidden state at current time-step

        self.x = x
        self.hidden = h

        # Add your code here.
        # Define your variables based on the writeup using the corresponding
        # names below.

        self.z = self.z_act.forward(np.matmul(self.Wzh, self.hidden) + np.matmul(self.Wzx, self.x))
        self.r = self.r_act.forward(np.matmul(self.Wrh, self.hidden) + np.matmul(self.Wrx, self.x))
        self.h_tilda = self.h_act.forward(np.matmul(self.Wh, np.multiply(self.r, self.hidden)) + np.matmul(self.Wx, self.x))
        self.h_t = np.multiply(1 - self.z, self.hidden) + np.multiply(self.z, self.h_tilda)

        assert self.x.shape == (self.d,)
        assert self.hidden.shape == (self.h,)

        assert self.r.shape == (self.h,)
        assert self.z.shape == (self.h,)
        assert self.h_tilda.shape == (self.h,)
        assert self.h_t.shape == (self.h,)

        return self.h_t

    # This must calculate the gradients wrt the parameters and return the
    # derivative wrt the inputs, xt and ht, to the cell.
    def backward(self, delta):
        # input:
        #  - delta:  shape (hidden dim), summation of derivative wrt loss from next layer at
        #            the same time-step and derivative wrt loss from same layer at
        #            next time-step
        # output:
        #  - dx: Derivative of loss wrt the input x
        #  - dh: Derivative  of loss wrt the input hidden h

        # 1) Reshape everything you saved in the forward pass.
        # 2) Compute all of the derivatives
        # 3) Know that the autograders the gradients in a certain order, and the
        #    local autograder will tell you which gradient you are currently failing.

        dh_tilda = delta * self.z
        # dh_bar is derivative w.r.t whole input to tanh() gate
        dh_bar = dh_tilda * self.h_act.derivative()

        dr = self.hidden.reshape(1, -1) * np.matmul(dh_bar, self.Wh)
        # dr_bar is derivative w.r.t whole input to reset gate
        dr_bar = dr * self.r_act.derivative()

        dz = delta * (self.h_tilda - self.hidden)
        # dz_bar is derivative w.r.t whole input to update gate
        dz_bar = dz * self.z_act.derivative()

        self.dWrx += np.matmul(self.x.reshape(1, -1).T, dr_bar).T
        self.dWrh += np.matmul(self.hidden.reshape(1, -1).T, dr_bar).T

        self.dWzx += np.matmul(self.x.reshape(1, -1).T, dz_bar).T
        self.dWzh += np.matmul(self.hidden.reshape(1, -1).T, dz_bar).T

        self.dWx += np.matmul(self.x.reshape(1, -1).T, dh_bar).T
        self.dWh += np.matmul(np.multiply(self.r.reshape(1, -1), self.hidden.reshape(1, -1)).T, dh_bar).T

        dh = delta * (1. - self.z.reshape(1, -1)) + np.matmul(dr_bar, self.Wrh) + np.matmul(dz_bar, self.Wzh) + np.multiply(self.r.reshape(1, -1), np.matmul(dh_bar, self.Wh))
        dx = np.matmul(dr_bar, self.Wrx) + np.matmul(dz_bar, self.Wzx) + np.matmul(dh_bar, self.Wx)

        assert dx.shape == (1, self.d)
        assert dh.shape == (1, self.h)

        return dx, dh
