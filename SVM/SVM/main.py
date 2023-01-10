import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from algorithm import SVM

# click run to try

df = pd.read_csv('./winequality-red.csv', sep = ';')
data = np.array(df)

X = data[:, :-1]
Y = data[:, -1].T

dmin = np.min(X)
dmax = np.max(X)
X = (X - dmin) / (dmax - dmin)
Y = np.where(Y >= 6, 1, -1)

# linear (polyd = 1) kernel
print("Calculating..., kernel: linear")
mySVM = SVM(X, Y, C = 10, ker = 'poly', kpara = 1)
mySVM.fit()
preY = mySVM.Inference(X)
print('linear acc:', 1 - len(np.nonzero(preY - Y)[0]) / Y.shape[0])
print('nums of support vectors: ', len(np.nonzero(mySVM.alpha == 0)[0]) )

# RBF kernel
print("Calculating..., kernel: RBF")
mySVM = SVM(X, Y, C = 10, ker = 'gaussian', kpara = 0.02)
mySVM.fit()
preY = mySVM.Inference(X)
print('RBF acc:', 1 - len(np.nonzero(preY - Y)[0]) / Y.shape[0])
print('nums of support vectors: ', len(np.nonzero(mySVM.alpha == 0)[0]) )


K_FOLD = 5
div = np.linspace(0, X.shape[0], K_FOLD+1, dtype = int)

def acc_para(ktype, k_list):

    tr_acc_list = []
    te_acc_list = []

    for d in k_list:
        print(f"Calculating {d}...")
        train_acc = 0
        test_acc = 0

        # K-Fold cross validation
        for i, j in zip(div[:-1], div[1:]):

            trainX = np.delete(X, slice(i,j), axis = 0)
            trainY = np.delete(Y, slice(i,j), axis = 0)
            testX = X[i:j]
            testY = Y[i:j]

            mySVM = SVM(trainX, trainY, 10, ker = ktype, kpara = d)
            # mySVM = SVM(trainX, trainY, d, ker = ktype, kpara = 0.02)
            mySVM.fit()

            preY_train = mySVM.Inference(trainX)
            preY_test = mySVM.Inference(testX)

            train_acc += 1 - len(np.nonzero(preY_train - trainY)[0]) / trainY.shape[0]
            test_acc += 1 - len(np.nonzero(preY_test - testY)[0]) / testY.shape[0]
    
        train_acc = train_acc / K_FOLD
        test_acc = test_acc / K_FOLD
    
        print(train_acc, test_acc)
        tr_acc_list.append(train_acc)
        te_acc_list.append(test_acc)
    
    # change settings and uncomment to plot
    #fig = plt.figure(figsize = (10,6))
    #ax1 = fig.add_subplot(111)
    #ax1.set_ylabel('train acc')
    #ax1.set_xlabel('sigma')
    #ax1.set_xscale('log')
    #ax2 = ax1.twinx()
    #ax2.set_ylabel('test acc')
    #ax1.plot(k_list, tr_acc_list, color = 'blue')
    #ax2.plot(k_list, te_acc_list, color = 'red')
    #ax1.scatter(k_list, tr_acc_list, color = 'blue')
    ##ax2.scatter(k_list, te_acc_list, color = 'red')
    ##plt.savefig(f'./result/{ktype}_kpara_precise.jpg')
    #plt.savefig(f'./result/{ktype}_kpara.jpg')

# 5.1 kernel and accuracy
#acc_para('poly', np.arange(1,5,0.2))
#acc_para('gaussian', [0.001, 0.01, 0.1, 1, 5, 10, 20, 50])
#acc_para('gaussian', np.arange(0.01, 0.1, 0.01))

# 5.2 C and accuracy
#acc_para('gaussian', [0.01, 0.1, 1, 5, 10, 20, 50])