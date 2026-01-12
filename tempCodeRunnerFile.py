import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# 1. Create a Synthetic Dataset (For demonstration)
# In your real project, you would use: data = pd.read_csv('your_dataset.csv')
data = {
    'Annual_Income': [50000, 30000, 100000, 20000, 80000, 45000, 25000, 90000],
    'Age': [25, 45, 35, 20, 50, 23, 40, 33],
    'Loan_Amount': [5000, 10000, 20000, 2000, 15000, 4000, 7000, 12000],
    'Prior_Default': ['No', 'Yes', 'No', 'Yes', 'No', 'No', 'Yes', 'No'],
    'Credit_Score': [700, 550, 800, 450, 750, 680, 500, 720],
    'Outcome': [1, 0, 1, 0, 1, 1, 0, 1]  # 1 = Creditworthy, 0 = High Risk
}

df = pd.DataFrame(data)

# 2. Data Preprocessing
# Encode categorical data ('Prior_Default') into numbers
le = LabelEncoder()
df['Prior_Default'] = le.fit_transform(df['Prior_Default'])

# 3. Define Features (X) and Target (y)
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# 4. Split the data into Training and Testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Build and Train the Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 6. Make Predictions
y_pred = model.predict(X_test)

# 7. Evaluate the Model
print("--- Model Performance ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Example: Predict for a new customer
new_customer = [[60000, 29, 8000, 0, 710]] # Income, Age, Loan, Default(No=0), Score
prediction = model.predict(new_customer)
status = "Creditworthy" if prediction[0] == 1 else "High Risk"
print(f"\nNew Customer Status: {status}")