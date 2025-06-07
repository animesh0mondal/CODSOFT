# Customer Churn Prediction
This project predicts customer churn using machine learning models based on the 'Churn_Modelling.csv' dataset. 
It implements three algorithms—Logistic Regression, Random Forest, and Gradient Boosting—and evaluates their performance using accuracy and F1-score. 
The script generates a single figure with three side-by-side confusion matrices to visualize model performance.

## Project Overview
The goal is to predict whether a customer will churn ('Exited=1') or not ('Exited=0') using features like 'CreditScore', 'Age', 'Tenure', 'Balance', 'Geography', and 'Gender'.
The script preprocesses the data, trains the models, and displays results in a table and confusion matrix plots.

### Features
- Models: Logistic Regression, Random Forest, Gradient Boosting.
- Metrics: Accuracy and F1-score.
- Visualization: Three confusion matrices displayed side by side.
- 
## Requirements
- Python 3.6 or higher
- Libraries:
  - 'pandas'
  - 'scikit-learn'
  - 'matplotlib'
  - 'seaborn'

## Clone the Repository:
   '''bash
   git clone https://github.com/your-username/customer-churn-prediction.git
   cd customer-churn-prediction
