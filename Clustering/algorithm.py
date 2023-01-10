import numpy as np

class GuassianModel:

    def __init__(self, data, num_class, max_iter = 30):
        self.data = np.array(data)
        self.num_class = num_class
        # distribution of P(Z)
        self.Z_pdf = np.ones(num_class) / num_class
        # distribution of P(X|Z)
        self.X_Z_pdf = []
        self.iter = max_iter


    def Solve(self):
        self._Init_para()
        while self.iter > 0:
            posterior = self._Estep()
            self._Mstep(posterior)
            print('Solving..., now_iter: ', self.iter)

    def Inference(self, data = None, prob = False):
        result = []

        if data is None:
            data = self.data
        if len(data.shape) == 1:
            data = [data]

        for x in data:
            pdf = self._Posterior(x)
            if not prob:
                result.append(np.argmax(pdf))
            else:
                result.append(pdf)   # return P(Z|X) if prob is True

        return np.array(result)

    # return probability of Guassian Distribution
    def GuassianProb(self, mean, cov, x):
        cov = cov.astype(np.float64)
        det = np.linalg.det(cov) ** 0.5
        pcs = np.linalg.inv(cov)
        coeff = (2 * np.pi) ** (len(x)/2)
        coeff = 1 / (coeff * det)

        x_mu = np.array([x - mean])
        tmp = np.dot(x_mu, pcs)
        tmp = np.dot(tmp, x_mu.T)[0][0]

        prob = coeff * np.exp(-0.5 * tmp)
        return prob

    # init mean and covariance randomly
    def _Init_para(self):        
        xmin = np.min(self.data)
        xmax = np.max(self.data)

        for i in range(self.num_class):
            rd = np.random.random(self.data.shape[1])
            mean = xmin + rd * (xmax - xmin)
            cov = np.eye(self.data.shape[1])
            self.X_Z_pdf.append([mean, cov])

    # calculate posterior, P(Z|X)
    def _Posterior(self, x):
        pdf = np.ones(self.num_class)

        for i in range(self.num_class):
            mean = self.X_Z_pdf[i][0]
            cov = self.X_Z_pdf[i][1]
            pdf[i] = self.Z_pdf[i] * self.GuassianProb(mean, cov, x)

        # deal with possible underflow
        if np.sum(pdf) > 1e-30:
            pdf = pdf / np.sum(pdf)
        else:
            pdf = np.ones(self.num_class) / self.num_class

        return pdf

    def _Estep(self):
        posts = []
        for x in self.data:
            pdf = self._Posterior(x)
            posts.append(pdf)
        return np.array(posts)

    def _Mstep(self, posts):
        err = 0
        psum = np.sum(posts, axis = 0)

        for i in range(self.num_class):
            # update mean and cov accroding to formula
            mean = np.sum(posts[:,i].reshape(-1,1) * self.data, axis = 0) / psum[i]
            x_mu = self.data - mean
            cov = np.dot(posts[:,i] * x_mu.T, x_mu) / psum[i]

            err += np.linalg.norm(mean - self.X_Z_pdf[i][0])
            # update distribution
            self.Z_pdf[i] = psum[i] / self.data.shape[0]
            self.X_Z_pdf[i] = [mean, cov]

        if err < 1e-3:
            self.iter = 0
        else:
            self.iter -= 1




