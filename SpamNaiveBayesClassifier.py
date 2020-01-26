import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt

#  Read data from CSV file
span_data_frame = pd.read_csv("dataset/spam.csv", delimiter=',', encoding='UTF-8')
print(span_data_frame.head())
print(span_data_frame.describe())

# Drop the columns which are not required
# In our case Unnamed: 2, Unnamed: 3, Unnamed:4 are no longer required
span_data_frame.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
span_data_frame.info()

# Plot data
plt.xlabel("Labels")
plt.title('Number of ham and spam messages')

sb.countplot(span_data_frame.v1)

plt.show()

# Split the data into training and test set

X_train, X_test, Y_train, Y_test = train_test_split(span_data_frame.v2, span_data_frame.v1, test_size=0.15)

# gnb = GaussianNB()
# y_pred = gnb.fit(X_train, Y_train).predict(X_test)
# print(y_pred)
# print("Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0], (Y_test != y_pred).sum()))
