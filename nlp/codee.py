from time import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Load the data
df = pd.read_csv('Language Detection.csv')

# Using a smaller subset of data for faster testing
df = df.sample(frac=0.1, random_state=1234)

# Preprocessing our data
tfidf = TfidfVectorizer(max_features=500)  # Reduce the number of features
X = tfidf.fit_transform(df['Text']).toarray()
y = df['Language']

# Encode target classes
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Splitting the data
X_train, X_test, y_train_encoded, y_test_encoded = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Training the Logistic Regression model
logistic_model = LogisticRegression(max_iter=1000, random_state=1234)
logistic_model.fit(X_train, y_train_encoded)

# Getting predictions from the Logistic Regression model
logistic_predictions_train = logistic_model.predict(X_train)
logistic_predictions_test = logistic_model.predict(X_test)

# Append predictions from Logistic Regression as new features
X_train_augmented = np.hstack((X_train, logistic_predictions_train.reshape(-1, 1)))
X_test_augmented = np.hstack((X_test, logistic_predictions_test.reshape(-1, 1)))

# Training the XGBoost model using the augmented data
xgb_model = XGBClassifier(n_estimators=100, max_depth=3, random_state=1234)
xgb_model.fit(X_train_augmented, y_train_encoded)

# Calculate accuracy on the test set
xgb_predictions_test = xgb_model.predict(X_test_augmented)
accuracy = accuracy_score(y_test_encoded, xgb_predictions_test)
print("Accuracy of the XGBoost model:", accuracy)

# User interaction for prediction
while True:
    user_input = input("Enter text to predict its language (type 'exit' to quit): ")
    if user_input.lower() == 'exit':
        print("Exiting...")
        break
    else:
        # Preprocess user input
        user_input_vectorized = tfidf.transform([user_input]).toarray()
        
        # Get prediction from Logistic Regression model
        logistic_prediction = logistic_model.predict(user_input_vectorized)
        
        # Append Logistic Regression prediction as new feature
        user_input_augmented = np.hstack((user_input_vectorized, logistic_prediction.reshape(-1, 1)))
        
        # Get prediction from XGBoost model using augmented features
        xgb_prediction = xgb_model.predict(user_input_augmented)
        
        # Decode the predicted class label
        predicted_language = label_encoder.inverse_transform(xgb_prediction)
        
        # Display predictions
        print("Predicted Language:", predicted_language[0])
