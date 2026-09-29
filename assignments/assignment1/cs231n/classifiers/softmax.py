from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


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

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    for i in range(num_train):
        scores = X[i].dot(W)

        # compute the probabilities in numerically stable way
        scores -= np.max(scores)
        p = np.exp(scores)
        p /= p.sum()  # normalize
        logp = np.log(p)
        
        loss -= logp[y[i]]  # negative log probability is the loss

        # 将正确类别的概率减 1，得到 (p - one_hot(y))，这是 Softmax 梯度的数学结果
        p[y[i]] -= 1
        # 计算梯度贡献, 当前样本特征向量的外积，并累加到 dW 中
        dW += np.outer(X[i], p)

    # normalized hinge loss plus regularization
    loss = loss / num_train + reg * np.sum(W * W)

    # 对梯度进行归一化，并加上正则化项的梯度 2 * reg * W
    dW = dW / num_train + 2 * reg * W

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################


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
    # TODO:                                                                     #
    # Implement a vectorized version of the softmax loss, storing the           #
    # result in loss.                                                           #
    #############################################################################

    num_train = X.shape[0]
    
    # 计算所有样本的分数矩阵 (N, C)
    scores = X.dot(W)
    
    # 减去每行的最大值，keepdims=True 确保形状为 (N, 1) 以便广播
    scores -= np.max(scores, axis=1, keepdims=True)
    
    # 计算 Softmax 概率矩阵
    exp_scores = np.exp(scores)
    p = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) # (N, C)
    
    # 计算损失
    # 提取每个样本对应正确类别的概率，取对数后求平均，最后加上正则化项
    loss = -np.sum(np.log(p[np.arange(num_train), y])) / num_train + reg * np.sum(W * W)

    # 使用向量化操作计算梯度
    # 复制概率矩阵，并对每个样本的正确类别位置减1 (P - Y)
    dscores = p.copy()
    dscores[np.arange(num_train), y] -= 1
    
    # 归一化
    dscores /= num_train
    
    # 权重梯度 = X^T * dscores + 正则化梯度
    dW = X.T.dot(dscores) + 2 * reg * W

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the softmax            #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################


    return loss, dW
