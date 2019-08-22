import dataset
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
plt.rcParams['figure.figsize'] = (7.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# 加载数据集
train_x, train_y, test_x, test_y = dataset.load_dataset(is_plot=True)

# (x, y, learning_rate=0.3, iterations=30000, print_cost=True, is_plot=True, lamdb=0, keep_prob=1)
parameters = dataset.model(train_x, train_y, learning_rate=0.3, iterations=30000, print_cost=True,keep_prob=0.85, is_plot=True,  plot_boundary=False)


accuracy_train = dataset.predict(train_x, train_y, parameters)
accuracy_test = dataset.predict(test_x, test_y, parameters)
print("训练集：{0}   验证集： {1}".format(accuracy_train, accuracy_test))


