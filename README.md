# Chatbot Project

## Overview

This project demonstrates how to build a chatbot using **Rasa**, **TF-IDF**, and **Logistic Regression** for intent classification. It processes user inputs to determine their intents and responds appropriately.

## Prerequisites

- Python 3.x
- Rasa framework
- Required Python libraries (listed in `requirements.txt`)

### Install Dependencies

```bash
# Clone the repo and navigate to the project directory
git clone https://github.com/yourusername/chatbot-project.git
cd chatbot-project

# Install necessary dependencies
pip install -r requirements.txt
```

## Dataset

The project uses a dataset of user queries (`utterances`) and their corresponding intents. The dataset is stored in `data/training_dataset.csv`.

## Usage

### 1. Train the Model (TF-IDF)

```python
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(df.utterance, df.label_num, test_size=0.2, stratify=df.label_num)

# Create pipeline and train
clf_tfid = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('logreg', LogisticRegression(C=1.0, penalty='l2', max_iter=100))
])
clf_tfid.fit(X_train, y_train)

# Model evaluation
y_pred = clf_tfid.predict(X_test)
print(classification_report(y_test, y_pred))
```

### 2. Rasa Chatbot

To train and test the Rasa-based chatbot:
```bash
rasa train
rasa shell
```

## License

This project is licensed under the MIT License.
