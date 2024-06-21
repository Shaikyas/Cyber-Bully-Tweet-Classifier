# Cyber-Bully-Tweet-Classifier Intro
A Cyber Bully Tweet Classifier is a machine learning model designed to identify and categorize tweets as either bullying or non-bullying.
It uses natural language processing (NLP) techniques to analyze the content of tweets.
The classifier is trained on labeled datasets containing examples of both bullying and non-bullying tweets.
By learning patterns and features from this data, it can predict the likelihood that a new tweet contains cyberbullying. 
This tool helps in monitoring and mitigating online harassment.
Cyberbullying on Twitter involves using tweets to harass, threaten, or demean individuals, often under the veil of anonymity. 
This form of digital harassment can quickly reach a wide audience due to the platform's public and rapid nature. 
Common tactics include insults, spreading rumors, defamatory content, and inciting others to participate in the harassment. 
The consequences for victims are severe, often leading to emotional distress, anxiety, depression, and even suicidal thoughts.
# Importing lybraries
import pandas as pd
import numpy as np
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
# Load and Preprocess Data
# Load dataset
# Assuming a dataset with columns 'text' for tweet and 'label' for classification (1 for bullying, 0 for non-bullying)
data = pd.read_csv('cyberbullying_tweets.csv')
# Preprocessing function
def preprocess_text(text):
    text = text.lower() # Lowercase text
    text = re.sub(r'http\S+', '', text) # Remove URLs
    text = re.sub(r'@\w+', '', text) # Remove mentions
    text = re.sub(r'#\w+', '', text) # Remove hashtags
    text = re.sub(r'[^\w\s]', '', text) # Remove punctuation
    text = re.sub(r'\d+', '', text) # Remove digits
    text = text.strip() # Remove leading and trailing spaces
    return text
# Apply preprocessing
data['text'] = data['text'].apply(preprocess_text)

# Split Data into Training and Test Sets
X = data['text']
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize Text Data
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train Logistic Regression Model
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# Evaluate Model
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1 Score: {f1:.2f}')

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Bullying', 'Bullying'], yticklabels=['Non-Bullying', 'Bullying'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()





