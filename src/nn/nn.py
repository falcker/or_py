import numpy as np
from numpy.random import randn

# init
N, Din, H, Dout = 2, 2, 2, 2

x = randn(N, Din)
y = randn(N, Dout)
w1, w2 = randn(Din, H), randn(H, Dout)
# loop
for t in range(10_000):
    # activation sigmoid                    # Forward pass
    h = 1.0 / (1.0 + np.exp(-x.dot(w1)))  #
    y_pred = h.dot(w2)  #
    # loss                                  #
    loss = np.square(y_pred - y).sum()  #

    # update                                # backprop
    dy_pred = 2.0 * (y_pred - y)  # Compute gradients
    dw2 = h.T.dot(dy_pred)
    dh = dy_pred.dot(w2.T)
    dw1 = x.T.dot(dh * h * (1 - h))

    w1 -= 1e-4 * dw1  # SGD step / descent
    w2 -= 1e-4 * dw2  #
