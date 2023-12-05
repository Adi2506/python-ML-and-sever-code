import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, auc
import seaborn as sns
import joblib

df = pd.read_csv('Spam.csv', index_col=None)

df = df[['CONTENT', 'CLASS']]

df['CLASS'] = df['CLASS'].map({0: "Not Spam", 1: "Spam Comment"})

x = np.array(df['CONTENT'])
y = np.array(df['CLASS'])

cv = CountVectorizer()
x = cv.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

nb = BernoulliNB()
nb.fit(x_train, y_train)
print(nb.score(x_test, y_test))

y_pred = nb.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

print(f'Accuracy : {accuracy}')
print('classification report:\n')
print(report)

# joblib.dump(nb, 'spam_model.pkl')
# joblib.dump(cv, 'spam_vectorizer.pkl')

# sample = "Check this out: https://thecleverprogrammer.com/"
# data = cv.transform([sample]).toarray()
# print(nb.predict(data))
