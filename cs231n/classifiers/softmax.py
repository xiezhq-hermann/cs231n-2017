import numpy as np
from random import shuffle


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    num_classes = W.shape[1]
    num_train = X.shape[0]
    scores = np.dot(X, W)
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################
    for i in range(num_train):
        score_i = scores[i, :] - np.max(scores[i, :])
        score_exp = np.exp(score_i)
        score_exp_sum = np.sum(score_exp)
        loss_i = -score_i[y[i]] + np.log(score_exp_sum)
        loss += loss_i

        for j in range(num_classes):
            gra_entropy = score_exp[j] / score_exp_sum
            if j == y[i]:
                dW[:, j] += (-1 + gra_entropy) * X[i]
            else:
                dW[:, j] += gra_entropy * X[i]
    loss /= num_train
    loss += reg * np.sum(W * W)
    dW /= num_train
    dW += 2 * reg * W
    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    num_train = X.shape[0]
    scores = np.dot(X, W)
    score_deduct = scores - np.max(scores, axis=1)[..., np.newaxis]
    score_exp = np.exp(score_deduct)
    score_exp_sum = np.sum(score_exp, axis=1)
    dScore = score_exp / score_exp_sum[..., np.newaxis]
    dScore[range(num_train), y] -= 1

    dW = np.dot(X.T, dScore)
    dW /= num_train
    dW += 2 * reg * W
    loss = -np.sum(np.log(np.choose(y, score_exp.T) / score_exp_sum.T))
    loss /= num_train
    loss += reg * np.sum(W * W)

    return loss, dW
