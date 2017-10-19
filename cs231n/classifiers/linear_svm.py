import numpy as np
from random import shuffle


def svm_loss_naive(W, X, y, reg):
	"""
	Structured SVM loss function, naive implementation (with loops).

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
	dW = np.zeros(W.shape)  # initialize the gradient as zero

	# compute the loss and the gradient
	num_classes = W.shape[1]
	num_train = X.shape[0]
	loss = 0.0

	for i in range(num_train):
		scores = X[i].dot(W)
		correct_class_score = scores[y[i]]
		count_above_margin = 0

		for j in range(num_classes):
			if j == y[i]:
				continue
			margin = scores[j] - correct_class_score + 1  # note delta = 1
			if margin > 0:
				count_above_margin += 1
				loss += margin
				dW[:, j] = dW[:, j] + X[i, :]

		dW[:, y[i]]= dW[:, y[i]] - X[i, :] * count_above_margin

	# Right now the loss is a sum over all training examples, but we want it
	# to be an average instead so we divide by num_train.
	loss /= num_train

	# Add regularization to the loss.
	loss += reg * np.sum(W * W)

	#############################################################################
	# TODO:                                                                     #
	# Compute the gradient of the loss function and store it dW.                #
	# Rather that first computing the loss and then computing the derivative,   #
	# it may be simpler to compute the derivative at the same time that the     #
	# loss is being computed. As a result you may need to modify some of the    #
	# code above to compute the gradient.                                       #
	#############################################################################
	dW = dW / num_train + 2 * reg * W

	return loss, dW


def svm_loss_vectorized(W, X, y, reg):
	"""
	Structured SVM loss function, vectorized implementation.

	Inputs and outputs are the same as svm_loss_naive.
	"""
	loss= 0.0
	dW= np.zeros(W.shape) # initialize the gradient as zero

	#############################################################################
	# TODO:                                                                     #
	# Implement a vectorized version of the structured SVM loss, storing the    #
	# result in loss.                                                           #
	#############################################################################
	scores = np.dot(X, W)
	correct_class_scores = np.choose(y, scores.T)

	scores_fil = scores.copy()
	scores_fil[range(scores.shape[0]), y] -= 1
	scores_margin = scores_fil - correct_class_scores[..., np.newaxis] + 1
	scores_margin[scores_margin < 0] = 0

	num_train = X.shape[0]
	loss = np.sum(scores_margin) / num_train
	loss += reg * np.sum(W * W)

	#############################################################################
	# TODO:                                                                     #
	# Implement a vectorized version of the gradient for the structured SVM     #
	# loss, storing the result in dW.                                           #
	#                                                                           #
	# Hint: Instead of computing the gradient from scratch, it may be easier    #
	# to reuse some of the intermediate values that you used to compute the     #
	# loss.                                                                     #
	#############################################################################
	epsilon = np.finfo(np.float).eps
	gra_indicator = (scores_margin > epsilon).astype(float)
	sum_ind = gra_indicator.sum(1)
	gra_indicator[range(gra_indicator.shape[0]), y] = -sum_ind
	dW = np.dot(X.T, gra_indicator)
	dW = dW / num_train + 2 * reg * W

	return loss, dW
