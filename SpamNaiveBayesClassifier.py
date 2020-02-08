import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

#  Read data from CSV file
span_data_frame = pd.read_csv("dataset/spam.csv", encoding='latin-1')
print(span_data_frame.head())
print(span_data_frame.describe())

# Drop the columns which are not required
# In our case Unnamed: 2, Unnamed: 3, Unnamed:4 are no longer required
span_data_frame.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
span_data_frame = span_data_frame.rename(columns={"v1": "class", "v2": "message"})
print(span_data_frame.head())
span_data_frame.info()

# Ploat data to display in graphical view
plt.xlabel("Labels")
plt.title('Number of ham and spam messages')

sb.countplot(span_data_frame['class'])

plt.show()

# Split the data into training and test set

X_train, X_test, Y_train, Y_test = train_test_split(span_data_frame['message'], span_data_frame['class'],
                                                    test_size=0.15)

# Transform data
# TfidfVectorizer coverts a raw data to Matrix

vectorizer = TfidfVectorizer()

X_train_transformed = vectorizer.fit_transform(X_train).toarray()
X_test_transformed = vectorizer.transform(X_test).toarray()


# Predicting the model and calculating model accuracy score
try:

    gnb = GaussianNB()

    gnb.fit(X_train_transformed, Y_train)

    model_prediction = gnb.predict(X_test_transformed)
    model_accuracy_score = accuracy_score(Y_test, model_prediction)

    print(model_prediction)

    print("Model accuracy score is %f" % model_accuracy_score)

except Exception as e:
    print(e)
