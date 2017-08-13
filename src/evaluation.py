import json as j
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfTransformer


# --- read and transform json file
json_data = None
with open('../data/yelp_academic_dataset_review.json') as data_file:
    lines = data_file.readlines()
    joined_lines = "[" + ",".join(lines) + "]"

    json_data = j.loads(joined_lines)

data = pd.DataFrame(json_data)
print(data.head())


# --- prepare the data

data = data[data.stars != 3]
data['sentiment'] = data['stars'] >= 4
print(data.head())

# --- build the model

X_train, X_test, y_train, y_test = train_test_split(data, data.sentiment, test_size=0.2)

# -
count = CountVectorizer()
temp = count.fit_transform(X_train.text)

tdif = TfidfTransformer()
temp2 = tdif.fit_transform(temp)

text_regression = LogisticRegression()
model = text_regression.fit(temp2, y_train)

prediction_data = tdif.transform(count.transform(X_test.text))

predicted = model.predict(prediction_data)


# --- make predictions

print("Total number of observations: " + str(len(y_test.values)))
print("Positives in observation: " + str(sum(y_test.values)))
print("Negatives in observation: " + str(len(y_test.values) - sum(y_test.values)))

print("Majority class is: " + str(sum(y_test.values)/len(y_test.values)*100) + "%")
print()

from sklearn.metrics import confusion_matrix

# Thus in binary classification, the count of

# true negatives is C_{0,0},
# false negatives is C_{1,0},
# true positives is C_{1,1} and
# false positives is C_{0,1}.

print("confusion matrix: \n" + str(confusion_matrix(y_test, predicted)))

from sklearn.metrics import accuracy_score

print()
print("accuracy: " + str(accuracy_score(y_test, predicted)))

from sklearn.metrics import precision_score

print()
print("precision: " + str(precision_score(y_test, predicted)))

from sklearn.metrics import recall_score

print()
print("recall: " + str(recall_score(y_test, predicted)))

from sklearn.metrics import f1_score

print()
print("f1 score: " + str(f1_score(y_test, predicted)))