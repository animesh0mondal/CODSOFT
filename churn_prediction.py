import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# Load the dataset
data = pd.read_csv(r'D:\customer_charm\Churn_Modelling.csv')

# Drop columns we don't need
data = data.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)

# Features (X) and target (y)
X = data.drop('Exited', axis=1)
y = data['Exited']

# Define number and category columns
number_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']
category_cols = ['Geography', 'Gender']

# Set up preprocessing
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), number_cols),
    ('cat', OneHotEncoder(drop='first'), category_cols)
])

# Split data into train (80%) and test (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models
models = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42)
}

# Store results
results = {}
confusion_matrices = {}

# Train and evaluate each model
for name, model in models.items():
    # Create pipeline
    pipeline = Pipeline([
        ('preprocess', preprocessor),
        ('model', model)
    ])
    
    # Train the model
    pipeline.fit(X_train, y_train)
    
    # Make predictions
    y_pred = pipeline.predict(X_test)
    
    # Calculate scores
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Store results
    results[name] = {'Accuracy': accuracy, 'F1 Score': f1}
    
    # Store confusion matrix
    confusion_matrices[name] = confusion_matrix(y_test, y_pred)
    
    # Print results
    print(f"\nResults for {name}:")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"F1 Score: {f1:.2f}")

# Create a figure with three subplots side by side
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Confusion Matrices for Churn Prediction Models', fontsize=16)

# Plot confusion matrix for each model
for idx, (name, cm) in enumerate(confusion_matrices.items()):
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=axes[idx])
    axes[idx].set_title(name)
    axes[idx].set_xlabel('Predicted')
    axes[idx].set_ylabel('Actual')
    axes[idx].set_xticks([0.5, 1.5])
    axes[idx].set_xticklabels(['Not Churned (0)', 'Churned (1)'])
    axes[idx].set_yticks([0.5, 1.5])
    axes[idx].set_yticklabels(['Not Churned (0)', 'Churned (1)'])

# Adjust layout to prevent overlap
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# Show results in a table
results_df = pd.DataFrame(results).T
print("\nModel Comparison:")
print(results_df)

# Pick the best model based on F1 Score
best_model = results_df['F1 Score'].idxmax()
print(f"\nBest Model: {best_model}")
print(f"Performance:\n{results_df.loc[best_model]}")