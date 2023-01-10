import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from algorithm import LinearRegression, LogisticRegression

K_FOLD = 5

df = pd.read_csv('./winequality-red.csv', sep = ';')
data = np.array(df)

# ---------- compute final para --------------
result = LinearRegression(data, None, 0.0075)
print('Linear(ridge) result:', result)
result = LogisticRegression(data, None, 1, 0)
print('Logistic result:', result)


# -------- linear/logistic lambda-error ---------
div = np.linspace(0, data.shape[0], K_FOLD+1, dtype = int)
def err_lambda(algorithm, rk_list):

    tr_err_list = []
    te_err_list = []

    for rk in rk_list:
        train_err = 0
        test_err = 0
        # K-fold cross validation
        for i, j in zip(div[:-1], div[1:]):
            train = np.delete(data, slice(i,j), axis = 0)
            test = data[i:j]
            if algorithm == 'linear':
                result = LinearRegression(train, test, rk)
            elif algorithm == 'logistic':
                result = LogisticRegression(train, test, 1, rk)
            train_err += result['train error']
            test_err += result['test error']
    
        train_err = train_err / K_FOLD
        test_err = test_err / K_FOLD
    
        tr_err_list.append(train_err)
        te_err_list.append(test_err)
    
    # plot
    fig = plt.figure(figsize = (10,6))
    ax1 = fig.add_subplot(111)
    ax1.set_xlabel('regularize lambda')
    ax1.set_ylabel('train error')
    ax1.plot(rk_list, tr_err_list, color = 'blue')
    ax2 = ax1.twinx()
    ax2.set_ylabel('test error')
    ax2.plot(rk_list, te_err_list, color = 'red')
    plt.savefig(f'./result/{algorithm}_lambda.jpg')
    plt.show()

#err_lambda('linear', np.arange(0, 1, 0.05))
print('Computing...')
err_lambda('linear', np.arange(0,0.2, 0.0001))
#err_lambda('logistic', np.arange(0, 5, 0.1))
print('Finish, result saved to ./result')


# -------- logistic alpha-iter_num -----------
# Uncomment to try
#lr_list1 = np.arange(0.01, 1, 0.1)
#lr_list2 = np.arange(1, 40, 2)
#lr_list = np.concatenate([lr_list1, lr_list2])

#plt.figure(figsize = (10,6))
#plt.ylabel('iter num')
#plt.xlabel('learning rate')

#iter_list = []
#for lr in lr_list:
#    result = LogisticRegression(data, None, lr, 0)
#    iter_list.append(result['iter num'])

#plt.plot(lr_list, iter_list)
#plt.savefig('./result/alpha_iter.jpg')
