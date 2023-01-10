import numpy as np

class DecisionTree:
    '''
    Discrete-feature data should be formatted in string.
    Decision Tree result is stored in a dict, keys:
      'idx': the index of feature which determines subtree
      'threshold': list for discrete feature and float for continuous
      'main': the most numerous feature (prepared for pruning)
      'next': subtrees
    Refer to _BuildTree() for more info.
    '''

    def __init__(self, data, label):
        self.X = np.array(data)
        self.Y = np.array(label)
        self.maxd = 15
        self.gini = False
        self.prune = False
        self.result = {}

    def Config(self, max_depth = 15, gini = False, prune = True):
        self.maxd = max_depth
        self.gini = gini
        self.prune = prune

    def BuildTree(self):
        self.result = self._BuildTree(self.X, self.Y, self.maxd)
        return self.result

    def Inference(self, data, label, tree):        
        err = 0
       
        for i in range(len(data)):
            result = tree
            while 'leaf' not in result.keys():       
                # select feature to divide
                idx = result['idx']
                # discrete feature (should be string)
                if isinstance(data[i][idx], str):
                    # select subtree
                    ptr = np.where(result['threshold'] == data[i][idx])[0]
                    result = result['next'][ptr]
                # continuous feature
                else:
                    if data[i][idx] < result['threshold']:
                        result = result['next'][0]
                    else:
                        result = result['next'][1]

            if result['leaf'] != label[i]:
                err += 1
                 
        return err
        

    def Back_pruning(self, ValData, ValLabel):
        self._Checknode(self.result, ValData, ValLabel)

    def Model_test(self, k_fold = 5):
        div = np.linspace(0, self.X.shape[0], k_fold+1, dtype = int)

        train_err = 0
        test_err = 0
        # K-fold cross validation
        for i, j in zip(div[:-1], div[1:]):
            train_X = np.delete(self.X, slice(i,j), axis = 0)
            train_Y = np.delete(self.Y, slice(i,j), axis = 0)
            test_X = self.X[i:j]
            test_Y = self.Y[i:j]

            if self.prune:
                tr_X, tr_Y, val_X, val_Y = self._GetVal(train_X, train_Y)
                self.result = self._BuildTree(tr_X, tr_Y, self.maxd)
                self.Back_pruning(val_X, val_Y)
            else:
                self.result = self._BuildTree(train_X, train_Y, self.maxd)

            train_err += self.Inference(train_X, train_Y, self.result) / train_X.shape[0]   
            test_err += self.Inference(test_X, test_Y, self.result) / (j-i)
                 

        train_acc = 1 - train_err / k_fold
        test_acc = 1 - test_err / k_fold
        return train_acc, test_acc

    def _BuildTree(self, data, label, now_depth):
        
        now_depth -= 1
        # get the most numerous label
        seq, counts = np.unique(label, return_counts = True)
        mainLabel = seq[np.argmax(counts)]

        if np.unique(label).shape[0] == 1 or len(data) == 0 or now_depth == 0:
            result = {'leaf': mainLabel}
            return result

        index, threshold = self._ChooseIndex(data, label)
                
        result = {'idx': index, 'threshold': threshold, 'main': mainLabel, 'next': []}
        splitData, splitLabel = self._SplitData(data, label, index, threshold)

        for subdata, sublabel in zip(splitData, splitLabel):
            result['next'].append( self._BuildTree(subdata, sublabel, now_depth) )

        return result


    def _Entropy(self, label):
        _, counts = np.unique(label, return_counts=True)
        entropy = 0
        if not self.gini:
            for cnt in counts:
                freq = cnt / label.shape[0]
                entropy -= freq * np.log2(freq)  # calculate entropy        
        else:
            for cnt in counts:
                freq = cnt / label.shape[0]
                entropy += freq * (1-freq)       # calculate gini index

        return entropy


    def _ChooseIndex(self, data, label):
        min_ent = 10
        index = 0
        threshold = None
        for i in range(data.shape[1]):

            col_i = data[:,i]
            entropy = 1
            thsld = None
 
            if isinstance(col_i[0], str):
                entropy = self._Entropy(label)
                thsld = np.unique(label)                
            else:
                entropy, thsld = self._seqEntropy(col_i, label)

            if entropy < min_ent:
                index = i
                min_ent = entropy
                threshold = thsld

        return index, threshold                         

    def _seqEntropy(self, data, label):
        min_ent = 10
        divth = 0
        seq = np.unique(data)  # sort and unique the continuous feature
        
        # traverse all sorted value
        for j in range(len(seq) -1):
            # determine the threshold for binarization
            div = (seq[j] + seq[j+1]) / 2
            idx1 = np.where(data < div)[0]
            idx2 = np.where(data >= div)[0]

            # calculate entropy
            freq = len(idx1) / data.shape[0]    
            entropy = freq * self._Entropy(label[idx1])  + (1-freq) * self._Entropy(label[idx2])

            # select the best threshold
            if (entropy < min_ent):
                min_ent = entropy
                divth = div

        return min_ent, divth


    def _SplitData(self, data, label, index, threshold):
        data_col = data[:, index]
        splitData = []
        splitLabel = []

        if isinstance(data_col[0], str):
            # for discrete feature, threshold is the list of classess
            for feature in threshold:
                idx = np.where(data_col == feature)[0]
                # eliminate discrete feature if explored
                new_data = np.delete(data[idx,:], index, axis = 1)
                splitData.append(new_data)
                splitLabel.append(label[idx])

        else:
            idx1 = np.where(data_col < threshold)
            idx2 = np.where(data_col >= threshold)

            splitData = [ data[idx1], data[idx2] ]
            splitLabel = [ label[idx1], label[idx2] ]

        return splitData, splitLabel


    def _Checknode(self, result, data, label):
        if 'leaf' in result.keys() or len(data) == 0:
            return

        # split validation data while recursing
        index = result['idx']
        threshold = result['threshold']
        splitData, splitLabel = self._SplitData(data, label, index, threshold)

        # recurse to check the lowest node first
        for i in range(len(splitData)):
            self._Checknode(result['next'][i], splitData[i], splitLabel[i])

        # calculate errors to determine if prune
        prun_err = 0
        for lb in label:
            if lb != result['main']: prun_err += 1               
 
        origin_err = self.Inference(data, label, result)

        if (prun_err < origin_err):
            result = {'leaf': result['main']}


    # divide validation set
    def _GetVal(self, data, label):
        index = {}
        for i in range(len(label)):
            if label[i] not in index.keys():
                index[label[i]] = [i]
            else:
                index[label[i]].append(i)

        train_idx = []
        val_idx = []

        for value in index.values():
            div = int(len(value)* 0.8)
            train_idx.extend(value[:div])
            val_idx.extend(value[div:])

        return data[train_idx], label[train_idx], data[val_idx], label[val_idx]




