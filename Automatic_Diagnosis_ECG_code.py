import numpy as np import pandas as pd import os
import os
import matplotlib.pyplot as plt import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split from sklearn.linear_model import LogisticRegression from sklearn.neighbors import KNeighborsClassifier from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score
from keras.utils import plot_model

for dirname, _, filenames in os.walk('/dataset_licenta'): for filename in filenames:
print(os.path.join(dirname, filename))



#Importing the csv data
AFdata = pd.read_csv('AFset.csv', header=None) NAFdata = pd.read_csv('NonAFset.csv', header=None)
print('Atrial Fibrilation heart data information') AFdata.info()
print ()
print('Non Atrial Fibrilation heart data information') NAFdata.info()
AFdata.head()
NAFdata.head()
_, (ax1,ax2) = plt.subplots(2,1, figsize=(16,9)) #Plotting the mean of the data
ax1.plot(AFdata.loc[:,:186].mean(), label='Atrial Fibrilation') ax1.plot(NAFdata.loc[:,:186].mean(), label='Non Atrial Fibrilation') ax1.set_title("Mean of the data points for heart data")
ax1.legend()


#Plotting the standard deviation for the data
ax2.plot(AFdata.loc[:,:186].std(), label='Atrial Fibrilation') ax2.plot(NAFdata.loc[:,:186].std(), label='Non Atrial Fibrilation') ax2.legend()
ax2.set_title("Standard Deviation for heart data")


#Concatenating the AF heart data with the Non AF one
print('All heart data information')
ALLdata = pd.concat([AFdata, NAFdata], axis=0, ignore_index=True) print(ALLdata.info())
ALLdata.head()


#Splitting into training and testing ALLdata.loc[:,187].value_counts()
X = ALLdata.loc[:,:186]
y = ALLdata.loc[:,187]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


#Logistic Regression
log_model = LogisticRegression(random_state=0, max_iter=5000) log_model.fit(X_train, y_train)
predicted_log=log_model.predict([X_test.iloc[0,:]]) print('Predict value using the LogisticRegression model:',end='') print(predicted_log)
print('The true value:',end='')
print(y_test.iloc[0])


#Decision Tree
dec_tree_model = DecisionTreeClassifier(random_state=0) dec_tree_model.fit(X_train, y_train)
predicted_tree = dec_tree_model.predict([X_test.iloc[1,:]]) print('predict value using the DecisionTreeClassifier model: ', end='') print(predicted_tree)
print('The true value: ',end='')
print(y_test.iloc[1])


#KNeighborsClassifier
knn_model = KNeighborsClassifier(n_neighbors=5) knn_model.fit(X_train, y_train)
predicted_knn = knn_model.predict([X_test.iloc[2,:]]) print('predict value using the knn model: ', end='')
print(predicted_knn) print('The true value: ',end='') print(y_test.iloc[2])


#Random Forest Classifier
rand_model = RandomForestClassifier() rand_model.fit(X_train, y_train)
predicted_rand = rand_model.predict([X_test.iloc[3,:]]) print('predict value using the Random Forest model: ', end='') print(predicted_rand)
print('The true value: ',end='')
print(y_test.iloc[3])


#Confusion Matrix for Logistic Regression
predicted = log_model.predict(X_test)
actual = y_test
confusion_matrix = metrics.confusion_matrix(actual, predicted)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = ["AF", "NonAF"])
cm_display.plot()
plt.title('Confusion Matrix for Logistic Regression')
plt.show()
#Confusion Matrix for Decision Tree
predicted = dec_tree_model.predict(X_test)
actual = y_test
confusion_matrix = metrics.confusion_matrix(actual, predicted)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = ["AF", "NonAF"])
cm_display.plot()
plt.title('Confusion Matrix for Decision Tree')
plt.show()
#Confusion Matrix KNN
predicted = knn_model.predict(X_test)
actual = y_test
confusion_matrix = metrics.confusion_matrix(actual, predicted)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = ["AF", "NonAF"])
cm_display.plot()
plt.title('Confusion Matrix for KNN')
plt.show()
#Confusion Matrix Random Forest
predicted = rand_model.predict(X_test)
actual = y_test
confusion_matrix = metrics.confusion_matrix(actual, predicted)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = ["AF", "NonAF"])
cm_display.plot()
plt.title('Confusion Matrix for Random Forest')
plt.show()


# Evaluate the logistic regression model predicted = log_model.predict(X_train) accuracy = accuracy_score(y_train, predicted) precision = precision_score(y_train, predicted) recall = recall_score(y_train, predicted)
f1 = f1_score(y_train, predicted)
print('Logistic Regression:') print(f'Accuracy: {accuracy:.4f}') print(f'Precision: {precision:.4f}') print(f'Recall: {recall:.4f}') print(f'F1-score: {f1:.4f}')
print()
# Evaluate the decision tree model
predicted = dec_tree_model.predict(X_train) accuracy = accuracy_score(y_train, predicted) precision = precision_score(y_train, predicted) recall = recall_score(y_train, predicted)
f1 = f1_score(y_train, predicted)
print('Decision Tree:') print(f'Accuracy: {accuracy:.4f}') print(f'Precision: {precision:.4f}') print(f'Recall: {recall:.4f}') print(f'F1-score: {f1:.4f}')
print()
# Evaluate the k-neighbors classifier
predicted = knn_model.predict(X_train) accuracy = accuracy_score(y_train, predicted) precision = precision_score(y_train, predicted) recall = recall_score(y_train, predicted)
f1 = f1_score(y_train, predicted)
print('K-Neighbors Classifier:') print(f'Accuracy: {accuracy:.4f}') print(f'Precision: {precision:.4f}') print(f'Recall: {recall:.4f}') print(f'F1-score: {f1:.4f}')
print()
# Evaluate the random forest classifier predicted = rand_model.predict(X_train) accuracy = accuracy_score(y_train, predicted) precision = precision_score(y_train, predicted) recall = recall_score(y_train, predicted)
f1 = f1_score(y_train, predicted)
print('Random Forest Classifier:') print(f'Accuracy: {accuracy:.4f}') print(f'Precision: {precision:.4f}') print(f'Recall: {recall:.4f}') print(f'F1-score: {f1:.4f}')



# Evaluation for Testing
# Calculate the sample weights
n_non_af = 10506
n_af = 4046
sample_weight = np.where(y_test == 0, 1 / n_non_af, 1 / n_af)
# Evaluate the logistic regression model predicted = log_model.predict(X_test) accuracy = accuracy_score(y_test, predicted) precision = precision_score(y_test, predicted) recall = recall_score(y_test, predicted)
f1 = f1_score(y_test, predicted)
ba=balanced_accuracy_score(y_test, predicted, sample_weight=sample_weight, adjusted=False)
print('Logistic Regression:') print(f'Accuracy: {accuracy:.4f}') print(f'Precision: {precision:.4f}') print(f'Recall: {recall:.4f}') print(f'F1-score: {f1:.4f}') print(f'Balanced Accuracy: {ba:.4f}') print()
# Evaluate the decision tree model

predicted = dec_tree_model.predict(X_test) accuracy = accuracy_score(y_test, predicted) precision = precision_score(y_test, predicted) recall = recall_score(y_test, predicted)
f1 = f1_score(y_test, predicted)
ba=balanced_accuracy_score(y_test, predicted, sample_weight=sample_weight, adjusted=False)
print('Decision Tree:') print(f'Accuracy: {accuracy:.4f}') print(f'Precision: {precision:.4f}') print(f'Recall: {recall:.4f}') print(f'F1-score: {f1:.4f}') print(f'Balanced Accuracy: {ba:.4f}') print()
# Evaluate the k-neighbors classifier
predicted = knn_model.predict(X_test)
accuracy = accuracy_score(y_test, predicted)
precision = precision_score(y_test, predicted)
recall = recall_score(y_test, predicted)
f1 = f1_score(y_test, predicted)
ba=balanced_accuracy_score(y_test, predicted, sample_weight=sample_weight, adjusted=False)
print('K-Neighbors Classifier:') print(f'Accuracy: {accuracy:.4f}') print(f'Precision: {precision:.4f}') print(f'Recall: {recall:.4f}') print(f'F1-score: {f1:.4f}') print(f'Balanced Accuracy: {ba:.4f}') print()
# Evaluate the random forest classifier predicted = rand_model.predict(X_test) accuracy = accuracy_score(y_test, predicted) precision = precision_score(y_test, predicted) recall = recall_score(y_test, predicted)
f1 = f1_score(y_test, predicted)
ba=balanced_accuracy_score(y_test, predicted, sample_weight=sample_weight, adjusted=False)
print('Random Forest Classifier:') print(f'Accuracy: {accuracy:.4f}') print(f'Precision: {precision:.4f}') print(f'Recall: {recall:.4f}') print(f'F1-score: {f1:.4f}') print(f'Balanced Accuracy: {ba:.4f}')



#CNN model
X_train = np.expand_dims(X_train, axis=2) X_test = np.expand_dims(X_test, axis=2)
cnn_model = Sequential()
cnn_model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(187, 1))) cnn_model.add(MaxPooling1D(pool_size=2))
cnn_model.add(Flatten())
cnn_model.add(Dense(50, activation='relu'))
cnn_model.add(Dense(1, activation='sigmoid'))
cnn_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) cnn_model.fit(X_train, y_train, epochs=10, batch_size=32)
loss, accuracy = cnn_model.evaluate(X_test, y_test) print(f'Test accuracy: {accuracy}')
plot_model(cnn_model, to_file='model.png', show_shapes=True, show_layer_names=True)



# Evaluate the CNN
predicted_probabilities = cnn_model.predict(X_test)
threshold = 0.5
predicted = (predicted_probabilities > threshold).astype(int).reshape(-1)
accuracy = accuracy_score(y_test, predicted)
precision = precision_score(y_test, predicted)
recall = recall_score(y_test, predicted)
f1 = f1_score(y_test, predicted)
balanced_accuracy = balanced_accuracy_score(y_test, predicted)
print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 score: {f1:.4f}')
print(f'Balanced accuracy: {balanced_accuracy:.4f}')