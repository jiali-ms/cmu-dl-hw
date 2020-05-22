# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np

class BatchNorm(object):

    def __init__(self, in_feature, alpha=0.9):

        # You shouldn't need to edit anything in init

        self.alpha = alpha
        self.eps = 1e-8
        self.x = None
        self.norm = None
        self.out = None

        # The following attributes will be tested
        self.var = np.ones((1, in_feature))
        self.mean = np.zeros((1, in_feature))

        self.gamma = np.ones((1, in_feature))
        self.dgamma = np.zeros((1, in_feature))

        self.beta = np.zeros((1, in_feature))
        self.dbeta = np.zeros((1, in_feature))

        # inference parameters
        self.running_mean = np.zeros((1, in_feature))
        self.running_var = np.ones((1, in_feature))

    def __call__(self, x, eval=False):
        return self.forward(x, eval)

    def forward(self, x, eval=False):
        """
        Argument:
            x (np.array): (batch_size, in_feature)
            eval (bool): inference status

        Return:
            out (np.array): (batch_size, in_feature)
        """

        if eval:
            out = (x - self.running_mean) / np.sqrt(self.running_var + self.eps)
            return self.gamma * out + self.beta
        else:
            self.x = x
            m = x.shape[0]

            self.mean = np.mean(x, axis=0)
            self.var = np.mean((x - self.mean) ** 2, axis=0)
            # B x I
            self.norm = (self.x - self.mean) * (self.var + self.eps) ** -0.5
            # B x I, 1 x I
            self.out = self.norm * self.gamma + self.beta

            self.running_mean = self.alpha * self.running_mean + (1 - self.alpha) * self.mean
            self.running_var = self.alpha * self.running_var + (1 - self.alpha) * self.var

            return self.out

    def backward(self, delta):
        """
        Argument:
            delta (np.array): (batch size, in feature)
        Return:
            out (np.array): (batch size, in feature)
        """
        # https://www.zhihu.com/question/38102762

        # B x I
        m = self.x.shape[0]

        dnorm = delta * self.gamma


        dvar = -0.5 * np.sum((dnorm * (self.x - self.mean) * (self.var + self.eps) ** -1.5), axis=0)
        dmean = -np.sum(dnorm * (self.var + self.eps) ** -0.5, axis=0) - 2 / m * dvar * np.sum(self.x - self.mean, axis=0)

        dx = dnorm * (self.var + self.eps) ** -0.5 + dvar * 2 * (self.x - self.mean) / m + dmean / m

        self.dbeta = np.mean(delta, axis=0, keepdims=True)
        self.dgamma = np.mean(delta * self.norm, axis=0, keepdims=True)

        return dx