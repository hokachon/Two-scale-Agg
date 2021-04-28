import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix



df = pd.read_csv("dataset2019.csv", usecols = ["label", "y_pred"])

label = df['label']
y_pred = df['y_pred']

conf_mat = confusion_matrix(label, y_pred)
ax = plt.subplot()
sn.set(font_scale=1.4) # for label size
sn.heatmap(conf_mat, annot=True, fmt="d", linewidths=.5, square=True, annot_kws={"size": 14}, cmap='Blues') # font size

ax.set_title('Confusion Matrix For 2019 dataset')
ax.set_xlabel('Predicted labels', fontsize = 18)
ax.set_ylabel('True labels', fontsize = 18)
ax.xaxis.set_ticklabels(['Airport', 'Bus', 'Metro', 'Metro_station', 'Park', 'Public_square',
                         'Shopping_mall', 'Street_pedestrian', 'Street_traffic', 'Tram'], fontsize = 18)

ax.yaxis.set_ticklabels(['Airport', 'Bus', 'Metro', 'Metro_station', 'Park', 'Public_square',
                         'Shopping_mall', 'Street_pedestrian', 'Street_traffic', 'Tram'], fontsize = 18)

plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.gcf().subplots_adjust(bottom=0.2)
plt.show()



# confusion matrix
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

df = pd.read_csv('dataset2019.csv', usecols=['label', 'y_pred'])
label = df['label']
y_pred = df['y_pred']


matrix = confusion_matrix(label, y_pred)
#print(matrix)
accuracy = matrix.diagonal() / matrix.sum(axis=1)
print(accuracy)

acc = np.sum(conf_mat.diagonal()) / np.sum(conf_mat)
print('Overall accuracy: {} %'.format(acc*100))
