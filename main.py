#Importing Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

#Loading Dataset
df = pd.read_csv("StudentsPerformance.csv")  
df.head()

#Data Preprocessing
print(df.info())
print(df.isnull().sum())  

# Converting categorical to numerical for ML
df['gender'] = df['gender'].map({'female': 0, 'male': 1})
df['race/ethnicity'] = df['race/ethnicity'].astype('category').cat.codes
df['parental level of education'] = df['parental level of education'].astype('category').cat.codes
df['lunch'] = df['lunch'].map({'standard': 1, 'free/reduced': 0})
df['test preparation course'] = df['test preparation course'].map({'none': 0, 'completed': 1})
df['average_score'] = df[['math score', 'reading score', 'writing score']].mean(axis=1)
df['passed'] = df['average_score'].apply(lambda x: 1 if x >= 50 else 0)

#Data Analysis & Visualization
sns.countplot(x='gender', hue='passed', data=df)
plt.title("Pass rate by Gender")
plt.show()

sns.boxplot(x='passed', y='average_score', data=df)
plt.title("Score distribution by Pass/Fail")
plt.show()

#  Correlation Heatmap
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()

#Machine Learning - Logistic Regression
features = ['gender', 'race/ethnicity', 'parental level of education', 'lunch',
            'test preparation course', 'math score', 'reading score', 'writing score']
X = df[features]
y = df['passed']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

#Evaluation
print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title("Confusion Matrix")
plt.show()
