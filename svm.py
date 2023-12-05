import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import joblib
from matplotlib import pyplot as plt
import seaborn as sns

df = pd.read_csv('train1.csv', encoding="ISO-8859-1", index_col=None)
df.dropna(subset=['selected_text'], inplace=True)


def preprocess_text(text):
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in string.punctuation]
    tokens = [word.lower() for word in tokens]

    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    cleaned_text = ' '.join(tokens)
    return cleaned_text


df['cleaned_text'] = df['text'].apply(preprocess_text)

sentiment_mapping = {
    'negative': 0,
    'neutral': 1,
    'positive': 2
}

df['sentiment_numeric'] = df['sentiment'].map(sentiment_mapping)

x = df['text']
y = df['sentiment_numeric']

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

svm_classifier = SVC(kernel='linear', C=1.0, random_state=42)
svm_classifier.fit(X_train_tfidf, y_train)

y_pred = svm_classifier.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

print('Classification Report:\n')
print(classification_report(y_test, y_pred))

confusion = confusion_matrix(y_test, y_pred)

# Display the confusion matrix
# print("Confusion Matrix:")
# print(confusion)
plt.figure(figsize=(8, 6))
sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

# joblib.dump(svm_classifier, 'svm_model.pkl')
# joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
