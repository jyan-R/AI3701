import numpy as np

class SVM:
    '''
    Implement of SVM using STO to solve the dual
    '''

    def __init__(self, X, Y, C, ker = 'poly', kpara = 1, max_iter = 200):
        self.X = np.array(X)
        self.Y = np.array(Y)
        self.C = C
        self.bia = 0
        self.weight = np.zeros([self.X.shape[1], 1])
        self.alpha = np.zeros(self.X.shape[0])
        self.ktype = ker
        self.kpara = kpara
        self.max_iter = max_iter

    def fit(self):
        self.kernel = self._GetKernel(self.X, self.X)
        iter_num = 0
        while iter_num < self.max_iter:
            cnt = 0
            iter_num += 1
            # rank indexs randomly
            idx = np.random.permutation(self.X.shape[0])
            # pick 2 dual variables to optimize
            for i, j in zip(idx[1:], idx[:-1]):
                cnt += self._Update(i, j)
            # break if all randomly selected alpha-pairs unchanged
            if cnt == 0 :
                break

    def Inference(self, test):
        X = np.array(test)
        test_ker = self._GetKernel(self.X, X)
        Y = [self.alpha *self.Y] @ test_ker + self.bia
        return np.where(Y[0]>0, 1, -1)
        
    def _Update(self, i, j):
        Y, K = self.Y, self.kernel
        dis = K[i][i] - 2*K[i][j] + K[j][j]
        g = lambda m: K[m] @ (self.alpha * Y)

        E_i = g(i) + self.bia - Y[i]
        E_j = g(j) + self.bia - Y[j]

        tmp = self.alpha[j] + Y[j] * (E_i - E_j) / (dis + 1e-10)    # new alpha unbounded
        new_j = self._Clip(i, j, tmp)   # clip alpha to fit the constraint

        delta_j = new_j - self.alpha[j]
        delta_i = -Y[i] *Y[j] *delta_j

        b1 = self.bia - E_i - Y[i] *K[i][i] *delta_i - Y[j] *K[i][j] *delta_j
        b2 = self.bia - E_j - Y[i] *K[i][j] *delta_i - Y[j] *K[j][j] *delta_j

        # update alpha and bia
        self.alpha[i] += delta_i
        self.alpha[j] += delta_j
        self.bia = 0.5 *(b1 + b2)

        if abs(delta_i) + abs(delta_j) < 1e-4:
            return 0        # alpha-pair unchanged
        else: return 1      # alpha-pair updated


    def _Clip(self, i, j, tmp):
        L, H = 0, 0
        C, Y, alpha = self.C, self.Y, self.alpha      

        if Y[i] != Y[j]:
            L = max(0, alpha[j] - alpha[i])
            H = min(C, C + alpha[j] - alpha[i])
        else:
            L = max(0, alpha[j] + alpha[i] - C)
            H = min(C, alpha[j] + alpha[i])

        return max(L, min(tmp, H))      # clip alpha in [L,H]

    def _GetKernel(self, X1, X2):
        
        if self.ktype == 'poly':
            d = self.kpara
            kernel = (X1 @ X2.T) ** d

        elif self.ktype == 'gaussian':
            sigma = self.kpara
            dis = np.sum( (X1[:, None, :] - X2) ** 2, axis = 2 )
            kernel = np.exp( -dis / (2 * sigma**2) )

        else:
            raise ValueError('only support poly or gaussian for kernel type!')

        return kernel
