import pandas as pd
import numpy as np
import time
from matplotlib import pyplot as plt
from algorithm import DecisionTree

# Just click run and get all the result

df = pd.read_table('./abalone.data',sep=',')

origin = np.array(df)
data = origin[:, :-1]
label = origin[:, -1]

# categorize labels to 0,1,2
for i, age in enumerate(label):
    if age <= 8 :
        label[i] = 0
    elif age == 9 or age == 10:
        label[i] = 1
    else:
        label[i] = 2
        
# create an instance
mytree = DecisionTree(data, label)

# 5.1 relationship between pruning and accuracy
train_list = []
test_list = []
train_listp = []
test_listp = []
config_list = []

# The process takes time due to the scale of data and iterative algorithm stragedy...
for depth in range(5,31,5):
    print('Building tree..., now depth: ', depth)
    mytree.Config(max_depth = depth, prune = False)
    train_err, test_err = mytree.Model_test()

    train_list.append(train_err)
    test_list.append(test_err)

    mytree.Config(max_depth = depth, prune = True)
    train_err, test_err = mytree.Model_test()

    train_listp.append(train_err)
    test_listp.append(test_err)
    config_list.append(depth)

plt.figure(figsize = (10,6))
plt.xlabel('max depth')
plt.ylabel('accuracy')
plt.ylim(0,1)
plt.plot(config_list, train_list, color = 'red', marker = '.')
plt.plot(config_list, train_listp, color = 'red', marker = '+')
plt.plot(config_list, test_list, color = 'blue', marker = '.')
plt.plot(config_list, test_listp, color = 'blue', marker = '+')
plt.legend(['train', 'pruned train', 'test', 'pruned test'])
plt.savefig(f'./result/synthesis.jpg')
plt.show()


# ----- uncomment to try -------
# 5.2 gini and entropy runtime contrast 
print('Comparing gini and entropy...')
mytree.Config(gini = False)
start = time.time()
mytree.BuildTree()
end = time.time()
print('entropy time:', end-start)

mytree.Config(gini = True)
start = time.time()
mytree.BuildTree()
end = time.time()
print('gini time:', end-start)

# 5.3 final result
print('Getting final accuracy...')
mytree.Config(max_depth = 10, gini = False, prune = True)
_, test_acc = mytree.Model_test()
print('Final accuracy:', test_acc)
