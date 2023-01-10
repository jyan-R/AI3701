import numpy as np
import functools


def LinearRegression(data, validate = None, rk = 0):

    data = np.array(data)
    X = data[:,:-1]
    Y = data[:, -1]
    
    # add 1 to data to get ready for matrix operation
    X = np.hstack([X, np.ones((X.shape[0], 1))])

    # analytical solotion
    pX = X.T @ X + rk * np.eye(X.shape[1])
    pX = np.linalg.inv(pX)
    weight = pX @ X.T @ Y

    # predict on training set and compute error
    train_err = np.linalg.norm(X @ weight.T - Y) / X.shape[0]
 
    # predict on test set
    test_err = None
    if validate is not None:
        validate = np.array(validate)
        testX = validate[:,:-1]
        testX = np.hstack([testX, np.ones((testX.shape[0], 1))])
        testY = validate[:, -1]

        test_err = np.linalg.norm(testX @ weight.T - testY) / testX.shape[0]

    # return result
    result = {'w': weight,
              'train error': train_err,
              'test error': test_err}

    return result


def LogisticRegression(data, validate = None, lr = 0.05, rk = 0, max_iter = 1000):

    data = np.array(data)
    X = data[:,:-1]
    Y = data[:,-1:]
    n_components, n_features = X.shape

    # binarization
    threshold = 0.6 * np.max(Y)
    Y = np.where(Y >= threshold, 1, 0)

    # normalization
    Xmin = np.min(X, axis = 0)
    Xmax = np.max(X, axis = 0)
    X = (X - Xmin) / (Xmax - Xmin)

    # initialize para for gradient desent
    weight = np.ones((n_features, 1))
    bia = 1
    wx_b = X @ weight + bia
    

    # define useful func
    #np_map = lambda f,x: np.array(list(map(f, x)))
    Sigmoid = lambda x: 1 / (1 + np.exp(-x))
    Ln_part = lambda x: np.log(1 + np.exp(x))

    # compute initial likelihood
    LLhood = np.mean(Y * wx_b - Ln_part(wx_b))
    LLhood += rk * np.sum(weight **2) / n_components  # regularization
  
    # apply gradient desent
    total_iter = max_iter
    for i in range(max_iter):
        
        dw = X.T @ (Sigmoid(wx_b) - Y) + rk * weight
        dw = dw / n_components
        db = np.sum(Sigmoid(wx_b) - Y) / n_components

        weight = weight - lr * dw
        bia = bia - lr * db
        
        wx_b = X @ weight + bia
        LLhood = np.mean(Y * wx_b - Ln_part(wx_b))
        LLhood += rk * np.sum(weight **2) / n_components

        # stop iter when gredient close to 0 
        if (np.linalg.norm(dw) + db * db < 1e-4):
            total_iter = i
            break

    # predict on training set
    prob = Sigmoid(X @ weight + bia)
    Y_pre = np.where(prob >= 0.5, 1, 0)
    train_err = np.linalg.norm(Y_pre - Y, ord=1)

    # predict on test set
    test_err = None
    if validate is not None:
        validate = np.array(validate)
        testX = validate[:,:-1]
        testY = validate[:, -1]

        testY = np.where(testY >= threshold, 1, 0)
        testX = (testX - Xmin) / (Xmax - Xmin)

        prob = Sigmoid(testX @ weight + bia)
        Y_pre = np.where(prob >= 0.5, 1, 0)
        test_err = np.linalg.norm(Y_pre - testY, ord=1)


    # return result
    result = {'w': weight, 
              'b': bia, 
              'train error': train_err,
              'test error': test_err,
              'likelihood': LLhood, 
              'iter num': total_iter}

    return result
