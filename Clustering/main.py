import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from algorithm import GuassianModel

df = pd.read_table('./abalone.data',sep=',')
origin = np.array(df)

# 5.1 Cluster directly
data = origin[:, 1:-1]
label = origin[:, -1]

# 5.2 Cluster seperately
# Uncomment to get result of Male and Female
#idx1 = np.where(origin[:,0]=='F')[0]   # Female
#idx1 = np.where(origin[:,0]=='M')[0]   # Male
#data = origin[idx1, 1:-1]
#label = origin[idx1, -1]

# normalize
xmin = np.min(data, axis = 0)
xmax = np.max(data, axis = 0)
data = (data - xmin) / (xmax - xmin)

for num_class in range(2,5):
    print(f'---- num_class: {num_class} -----')

    # create instance and solve
    gmm = GuassianModel(data, num_class, max_iter = 6)
    gmm.Solve()
    result = gmm.Inference()
    
    # synthesize results
    rscnt = np.zeros((num_class, 29))    
    for i in range(num_class):
        idx = np.where(result == i)[0]
        ages, cnts = np.unique(label[idx], return_counts = True)
        for age, cnt in zip(ages, cnts):
            rscnt[i, age-1] = cnt

    # plot
    plt.figure(figsize = (8,6))
    plt.xlabel('num', fontsize = 16)
    plt.ylabel('age', fontsize = 16)
    plt.plot(np.arange(1,30), rscnt.T)
    plt.savefig(f'./result/k_{num_class}.jpg')
    #plt.savefig(f'./result/Fk_{num_class}.jpg') # For Female
    #plt.savefig(f'./result/Mk_{num_class}.jpg') # For Male
    #plt.show()


















#plt.figure(figsize = (10,6))
#plt.xlabel('max depth')
#plt.ylabel('accuracy')
#plt.ylim(0,1)
#plt.plot(config_list, train_list, color = 'red', marker = '.')
#plt.plot(config_list, train_listp, color = 'red', marker = '+')
#plt.plot(config_list, test_list, color = 'blue', marker = '.')
#plt.plot(config_list, test_listp, color = 'blue', marker = '+')
#plt.legend(['train', 'pruned train', 'test', 'pruned test'])
#plt.savefig(f'./result/synthesis.jpg')
#plt.show()

