import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

train_file = pd.read_csv("fashion-mnist_train.csv")
test_file = pd.read_csv("fashion-mnist_test.csv")

#train_file.head()

#print(train_file.shape)
#print(test_file.shape)

y_train = train_file.values
y_train=np.delete(y_train,slice(1,785),axis=1)
y_train=y_train.ravel()

y_test = test_file.values
y_test=np.delete(y_test,slice(1,785),axis=1)
y_test=y_test.ravel()

#print(y_train.shape)
#print(y_test.shape)

#y_train[0:5]

train_file.drop('label',axis=1, inplace=True)
test_file.drop('label',axis=1, inplace=True)

#train_file.head()

train_file_arr = train_file.values

plt.figure(figsize=(20,4))
for index, (image, label) in enumerate(zip(train_file_arr[0:5], y_train[0:5])):
    plt.subplot(1, 5, index+1)
    plt.imshow(np.reshape(image,(28,28)), cmap=plt.cm.gray)
    plt.title('Training: %i\n' %label, fontsize = 20)

plt.show()

from sklearn.linear_model import LogisticRegression
logisticRegr = LogisticRegression(solver='lbfgs')
logisticRegr.fit(train_file,y_train)

#y_test[0:10]

#test_file_arr.shape

#test_file_arr = test_file.values


#logisticRegr.predict(test_file_arr[0].reshape(1,-1))

#logisticRegr.predict(test_file[0:10])

predictions=logisticRegr.predict(test_file)

score = logisticRegr.score(test_file, y_test)
print(score)

index=0
mis_classified_index = []
for label, predict in zip(y_test,predictions):
    if label!=predict:
        mis_classified_index.append(index)
    index+=1

plt.figure(figsize=(20,4))
for plot_index, bad_index in enumerate(mis_classified_index[0:5]):
    plt.subplot(1,5, plot_index+1)
    plt.imshow(np.reshape(test_file_arr[bad_index], (28,28)), cmap=plt.cm.gray)
    plt.title('Predicted: {}, Actual: {}'.format(predictions[bad_index], y_test[bad_index]), fontsize=20)
    
y_test[0:10]







