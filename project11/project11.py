# -*- coding: utf-8 -*-
# """Project11.ipynb

# Automatically generated by Colab.

# Original file is located at
#     https://colab.research.google.com/drive/1ZSSZIyptFRWSKXi-nvFqDa4v_12GIz93
# """

# Commented out IPython magic to ensure Python compatibility.
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pickle

import warnings
warnings.filterwarnings('ignore')

# Data Load
# Loading the dataset into a pandas DataFrame (assuming 'data.csv' is the dataset)
data=pd.read_csv('project11/loan_prediction.csv')
# data

# data.head()

# data.tail()

# data.describe()

# data.shape

# data.isna().sum()

# data.dtypes

# data.info()

# plt.figure(figsize=(20,15), facecolor='orange')
# plotnumber = 1

# for column in data.select_dtypes(include=['int64', 'float64']):  # Only select numeric columns
#     if plotnumber <= 9:
#         ax = plt.subplot(3, 3, plotnumber)
#         sns.distplot(data[column], kde=True)
#         plt.xlabel(column, fontsize=20)
#     plotnumber += 1

# plt.tight_layout()
# plt.show()

# plt.figure(figsize=(20, 15))
# plotnumber = 1

# # Plot numeric columns
# for column in data.select_dtypes(include=['int64', 'float64']):  # Only select numeric columns
#     if plotnumber <= 9:  # Limit to 9 plots
#         ax = plt.subplot(3, 3, plotnumber)
#         sns.histplot(data[column], kde=True)
#         plt.xlabel(column, fontsize=20)
#     plotnumber += 1

# plt.tight_layout()
# plt.show()

# plt.figure(figsize=(20, 10))

# # Columns to check for outliers in the loan dataset
# columns_to_check = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']

# for i, column in enumerate(columns_to_check, 1):
#     plt.subplot(2, 2, i)
#     sns.boxplot(data=data, x=column)
#     plt.title(f'Boxplot of {column}', fontsize=15)
#     plt.xlabel(column, fontsize=12)

# plt.tight_layout()
# plt.show()

# numeric_data = data.select_dtypes(include=['int64', 'float64'])
# numeric_data.corr()

# # Heatmap to check for correlations

# numeric_data = data.select_dtypes(include=['int64', 'float64'])
# correlation_matrix = numeric_data.corr()

# # Plot the heatmap
# plt.figure(figsize=(10, 8))
# sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
# plt.title('Correlation Matrix')
# plt.show()

# """## Preproccesing
# We will start with fixing the skewness that has been observed in the data
# """

# Apply log transformation to reduce skewness
data['ApplicantIncome'] = np.log1p(data['ApplicantIncome'])  # log1p to handle zero values
data['CoapplicantIncome'] = np.log1p(data['CoapplicantIncome'])
data['LoanAmount'] = np.log1p(data['LoanAmount'])

# plt.figure(figsize=(20,15), facecolor='orange')
# plotnumber = 1

# for column in data.select_dtypes(include=['int64', 'float64']):  # Only select numeric columns
#     if plotnumber <= 9:
#         ax = plt.subplot(3, 3, plotnumber)
#         sns.distplot(data[column], kde=True)
#         plt.xlabel(column, fontsize=20)
#     plotnumber += 1

# plt.tight_layout()
# plt.show()

# plt.figure(figsize=(20, 15))
# plotnumber = 1

# Plot numeric columns
# for column in data.select_dtypes(include=['int64', 'float64']):  # Only select numeric columns
#     if plotnumber <= 9:  # Limit to 9 plots
#         ax = plt.subplot(3, 3, plotnumber)
#         sns.histplot(data[column], kde=True)
#         plt.xlabel(column, fontsize=20)
#     plotnumber += 1

# plt.tight_layout()
# plt.show()

# """Now that the skewness has improved, we will go ahead and remove the outliers observed in the data."""

# Function to remove outliers using IQR
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# Remove outliers from specific columns
data = remove_outliers(data, 'ApplicantIncome')
data = remove_outliers(data, 'CoapplicantIncome')
data = remove_outliers(data, 'LoanAmount')

# plt.figure(figsize=(20, 10))

# # Columns to check for outliers in the loan dataset
# columns_to_check = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']

# for i, column in enumerate(columns_to_check, 1):
#     plt.subplot(2, 2, i)
#     sns.boxplot(data=data, x=column)
#     plt.title(f'Boxplot of {column}', fontsize=15)
#     plt.xlabel(column, fontsize=12)

# plt.tight_layout()
# plt.show()

# """Now that the outliers have been removed, we can go ahead to the next step, which is removing null values from the dataset."""

# Filling missing values in categorical columns with mode
data['Gender'].fillna(data['Gender'].mode()[0], inplace=True)
data['Married'].fillna(data['Married'].mode()[0], inplace=True)
data['Dependents'].fillna(data['Dependents'].mode()[0], inplace=True)
data['Self_Employed'].fillna(data['Self_Employed'].mode()[0], inplace=True)
data['Credit_History'].fillna(data['Credit_History'].mode()[0], inplace=True)

# Filling missing values in numeric columns
data['LoanAmount'].fillna(data['LoanAmount'].median(), inplace=True)
data['Loan_Amount_Term'].fillna(data['Loan_Amount_Term'].mode()[0], inplace=True)

# data.isnull().sum()

# """ We have removed the null values in the code. Now we can go ahead and remove any unneccessary columns in the dataset"""

# Dropping the Loan_ID column
data.drop(columns=['Loan_ID'], inplace=True)

# data.shape

# """Let's proceed with one-hot encoding for the categorical variables."""

from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

# Apply LabelEncoder to each categorical column
categorical_columns = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area', 'Loan_Status']
for column in categorical_columns:
    data[column] = label_encoder.fit_transform(data[column])

# data.head()

# """Here, we can see that for Loan Status, 0 stands for Yes and 1 stands for No

# ## Train Test Split


from sklearn.model_selection import train_test_split

# Define features (X) and target (y)
X = data.drop(columns=['Loan_Status'])
y = data['Loan_Status']

# Split the dataset into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Display the shapes of the training and test sets
# print("Training Features Shape:", X_train.shape)
# print("Test Features Shape:", X_test.shape)
# print("Training Labels Shape:", y_train.shape)
# print("Test Labels Shape:", y_test.shape)

# """Let's move on to Standardizing the data using Standard Scaler"""

from sklearn.preprocessing import StandardScaler

# Initializing the StandardScaler
scaler = StandardScaler()

# Fit the scaler on the training data and transform both the training and test sets
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# """### Defining a function that checks the model's Accuracy, MAE, MSE, R2 Score"""

# We have defined a metric method that prints the mae, mse and r2 score

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error, mean_squared_error, r2_score

def metrics_score(model, X_train, X_test, y_train, y_test, train=True):
    if train:
        y_pred = model.predict(X_train)
        print("==================Training Score=================")
        print("Accuracy ==> ", accuracy_score(y_train, y_pred))
        print("Mean Absolute Error (MAE) ==> ", mean_absolute_error(y_train, y_pred))
        print("Mean Squared Error (MSE) ==> ", mean_squared_error(y_train, y_pred))
        print("R-squared (R2) ==> ", r2_score(y_train, y_pred))
    else:
        y_pred = model.predict(X_test)
        print("==================Test Score=================")
        print("Accuracy ==> ", accuracy_score(y_test, y_pred))
        print("Mean Absolute Error (MAE) ==> ", mean_absolute_error(y_test, y_pred))
        print("Mean Squared Error (MSE) ==> ", mean_squared_error(y_test, y_pred))
        print("R-squared (R2) ==> ", r2_score(y_test, y_pred))

# """## 1st Model: Logistic Regression"""

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# Initialize Logistic Regression
logistic_reg = LogisticRegression()

# Fit the model on the training set
logistic_reg.fit(X_train, y_train)

# Print training metrics
# metrics_score(logistic_reg, X_train, X_test, y_train, y_test, train=True)

# # Print test metrics
# metrics_score(logistic_reg, X_train, X_test, y_train, y_test, train=False)

# Cross-validation (5-fold) to check for model robustness
cv_scores_logistic = cross_val_score(logistic_reg, X_train, y_train, cv=5, scoring='accuracy')
# print(f'Logistic Regression Cross-Validation Accuracy: {cv_scores_logistic.mean():.4f}')

# """## 2nd Model: Decision Tree"""

from sklearn.tree import DecisionTreeClassifier

# Initialize Decision Tree Classifier
decision_tree = DecisionTreeClassifier(random_state=42)

# Fit the model on the training set
decision_tree.fit(X_train, y_train)

# Print training metrics
# metrics_score(decision_tree, X_train, X_test, y_train, y_test, train=True)

# # Print test metrics
# metrics_score(decision_tree, X_train, X_test, y_train, y_test, train=False)

# Cross-validation (5-fold)
cv_scores_decision_tree = cross_val_score(decision_tree, X_train, y_train, cv=5, scoring='accuracy')
# print(f'Decision Tree Cross-Validation Accuracy: {cv_scores_decision_tree.mean():.4f}')

# """## 3rd Model: Random Forest"""

from sklearn.ensemble import RandomForestClassifier

# Initialize Random Forest Classifier
random_forest = RandomForestClassifier(random_state=42)

# Fit the model on the training set
random_forest.fit(X_train, y_train)

# # Print training metrics
# metrics_score(random_forest, X_train, X_test, y_train, y_test, train=True)

# # Print test metrics
# metrics_score(random_forest, X_train, X_test, y_train, y_test, train=False)

# Cross-validation (5-fold)
cv_scores_random_forest = cross_val_score(random_forest, X_train, y_train, cv=5, scoring='accuracy')
# print(f'Random Forest Cross-Validation Accuracy: {cv_scores_random_forest.mean():.4f}')

# """## 4th Model: K-Nearest Neighbours"""

from sklearn.neighbors import KNeighborsClassifier

# Initialize K-Nearest Neighbors Classifier
knn = KNeighborsClassifier()

# Fit the model on the training set
knn.fit(X_train, y_train)

# Print training metrics
# metrics_score(knn, X_train, X_test, y_train, y_test, train=True)

# # Print test metrics
# metrics_score(knn, X_train, X_test, y_train, y_test, train=False)

# Cross-validation (5-fold)
cv_scores_knn = cross_val_score(knn, X_train, y_train, cv=5, scoring='accuracy')
# print(f'K-Nearest Neighbors Cross-Validation Accuracy: {cv_scores_knn.mean():.4f}')

# """## 5th Model: MLP"""

from sklearn.neural_network import MLPClassifier

# Multilayer Perceptron Classifier
mlp = MLPClassifier()

# Fit the model on the training set
mlp.fit(X_train, y_train)

# Print training metrics
# metrics_score(mlp, X_train, X_test, y_train, y_test, train=True)

# # Print test metrics
# metrics_score(mlp, X_train, X_test, y_train, y_test, train=False)

# Cross-validation accuracy (5-fold)
cv_scores_mlp = cross_val_score(mlp, X_train, y_train, cv=5, scoring='accuracy')
# print(f'MLP Cross-Validation Accuracy: {cv_scores_mlp.mean():.4f}')

# """## 6th Model: SVM"""

from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

# Initialize Support Vector Classifier
svm_model = SVC(random_state=42)

# Fit the model on the training set
svm_model.fit(X_train, y_train)

# Print training metrics
# metrics_score(svm_model, X_train, X_test, y_train, y_test, train=True)

# # Print test metrics
# metrics_score(svm_model, X_train, X_test, y_train, y_test, train=False)

# Cross-validation (5-fold) to check for model robustness
cv_scores_svm = cross_val_score(svm_model, X_train, y_train, cv=5, scoring='accuracy')
# print(f'SVM Cross-Validation Accuracy: {cv_scores_svm.mean():.4f}')

# """| Model                | Training Accuracy | Test Accuracy | Cross-Validation Accuracy |
# |----------------------|-------------------|---------------|---------------------------|
# | Logistic Regression  | 83.49%           | 75.00%       | 83.26%                    |
# | Decision Tree        | 100%             | 71.30%       | 73.72%                    |
# | Random Forest        | 100%             | 72.22%       | 82.09%                    |
# | K-Nearest Neighbors  | 85.12%           | 73.15%       | 83.02%                    |
# | MLP                  | 86.28%           | 74.07%       | 81.40%                    |
# | SVM                  | 84.41%           | 75.00%       | 83.49%

# Logistic Regression and SVM performs the best on the test set and even after Cross Validating.

# ## Hyperparameter Tuning
# Let's perform Hyperparameter Tuning on the best model, which is SVM and Logistic Regression, and Random Forest, since it is close to SVM and performs well after hyperparameter tuning

# ### Hyperparameter Tuning for SVM
# """

from sklearn.svm import SVC

# Define parameter grid for SVM
param_grid_svm = {
    'C': [0.1,0.5, 1, 10, 100],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto']
}

# Initialize SVM and GridSearchCV
svm_model = SVC(random_state=42,class_weight='balanced')
grid_search_svm = GridSearchCV(estimator=svm_model, param_grid=param_grid_svm, cv=5, scoring='accuracy', n_jobs=-1, verbose=2)

# Fit GridSearchCV on SVM
grid_search_svm.fit(X_train, y_train)

# Best parameters and score for SVM
# print("Best Parameters for SVM:", grid_search_svm.best_params_)
# print("Best Cross-Validation Accuracy for SVM:", grid_search_svm.best_score_)

### Hyperparameter Tuning for Logistic Regression

# Define parameter grid for Logistic Regression
param_grid_logistic = {
    'C': [0.01, 0.1, 1, 10, 100],
    'solver': ['liblinear', 'saga']
}

# Initialize Logistic Regression and GridSearchCV
logistic_reg = LogisticRegression(random_state=42,class_weight='balanced')
grid_search_logistic = GridSearchCV(estimator=logistic_reg, param_grid=param_grid_logistic, cv=5, scoring='accuracy', n_jobs=-1, verbose=2)

# Fit GridSearchCV on Logistic Regression
grid_search_logistic.fit(X_train, y_train)

# Best parameters and score for Logistic Regression
# print("Best Parameters for Logistic Regression:", grid_search_logistic.best_params_)
# print("Best Cross-Validation Accuracy for Logistic Regression:", grid_search_logistic.best_score_)

## Hyperparameter Tuning for Random Forest

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Define parameter grid for Random Forest

param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Initialize Random Forest and GridSearchCV
random_forest = RandomForestClassifier(random_state=42, class_weight='balanced')
grid_search_rf = GridSearchCV(estimator=random_forest, param_grid=param_grid_rf, cv=5, scoring='accuracy', n_jobs=-1, verbose=2)

# Fit GridSearchCV on Random Forest
grid_search_rf.fit(X_train, y_train)

# Best parameters and score for Random Forest
# print("Best Parameters for Random Forest:", grid_search_rf.best_params_)
# print("Best Cross-Validation Accuracy for Random Forest:", grid_search_rf.best_score_)

# Evaluate the best model on the test set
best_rf_model = grid_search_rf.best_estimator_
test_accuracy = accuracy_score(y_test, best_rf_model.predict(X_test))
# print(f'Test Accuracy of the Best Random Forest Model: {test_accuracy:.4f}')

# """## Saving the 3 Models using Pickle"""

import pickle

# Save the best Logistic Regression model
with open('best_logistic_model.pkl', 'wb') as file:
    pickle.dump(grid_search_logistic.best_estimator_, file)

with open('best_random_forest_model.pkl', 'wb') as file:
    pickle.dump(grid_search_rf.best_estimator_, file)

with open('best_svm_model.pkl', 'wb') as file:
    pickle.dump(grid_search_svm.best_estimator_, file)

## Testing all 3 Models

# Example 1 explaination:
# *   A male applicant who is married, has no dependents, and is a graduate.
# *   Not self-employed with an income of 5000 and a coapplicant income of 2000.
# *   Requested a loan amount of 150 (in thousands) for a term of 360 months.
# *   Has a good credit history, and lives in an urban area.



# Example 2 explaination:


# *   A female applicant who is not married, with 1 dependent and not a graduate.
# *   Self-employed with an income of 3000, and no coapplicant income.
# *   Requested a loan of 100 (in thousands) for a term of 120 months.
# *   Has a poor credit history and lives in a rural area.

# Example 3 explaination:
# * A male applicant who is married, with 2 dependents, and is a graduate.
# * Not self-employed with a high income of 8000, plus a coapplicant income of 3000.
# * Requested loan amount of 200 (in thousands) with a 30-year term for manageable payments.
# * Good credit history and residence in an urban area, both favorable indicators.



# import pickle


# # [Gender, Married, Dependents, Education, Self_Employed, ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History, Property_Area]
# test_application_data = [
#     [1, 1, 0, 1, 0, 5000, 2000, 150, 360, 1, 2],  # Example 1
#     [0, 0, 1, 0, 1, 3000, 0, 100, 120, 0, 0],     # Example 2
#     [1, 1, 2, 1, 0, 8000, 3000, 200, 360, 1, 2]   # Example 3
# ]

# def print_predictions(model_name, predictions):
#     print(f"\n{model_name} Predictions for new examples:")
#     for i, prediction in enumerate(predictions, start=1):
#         status = "Loan Approved" if prediction == 1 else "Loan Not Approved"
#         print(f"Example {i}: {status}")


# # Load and test Logistic Regression model
# with open('best_logistic_model.pkl', 'rb') as model_file:
#     logistic_model = pickle.load(model_file)
# logistic_predictions = logistic_model.predict(test_application_data)
# print("Logistic Regression Predictions for the examples:", logistic_predictions)
# print_predictions("Logistic Regression", logistic_predictions,)

# # Load and test Random Forest model
# with open('best_random_forest_model.pkl', 'rb') as model_file:
#     random_forest_model = pickle.load(model_file)
# rf_predictions = random_forest_model.predict(test_application_data)
# print("Random Forest Predictions for the examples:", rf_predictions)
# print_predictions("Random Forest", rf_predictions)

# # Load and test SVM model
# with open('best_svm_model.pkl', 'rb') as model_file:
#     svm_model = pickle.load(model_file)
# svm_predictions = svm_model.predict(test_application_data)
# print("SVM Predictions for the examples:", svm_predictions)
# print_predictions("SVM", svm_predictions)

#### Therefore, we can see **Random Forest** performs the best and gives relatively more accurate predictions in comparison to Logistic Regression and SVMs.

# Deploying the Project


import streamlit as st
import numpy as np
import pickle

# Load all three models with updated filenames
with open('best_logistic_model.pkl', 'rb') as file:
    logistic_model = pickle.load(file)

with open('best_random_forest_model.pkl', 'rb') as file:
    random_forest_model = pickle.load(file)

with open('best_svm_model.pkl', 'rb') as file:
    svm_model = pickle.load(file)

# Streamlit app title
st.title("Loan Approval Prediction App")
st.write("Enter the applicant details below to predict if the loan would be approved by each model.")

# Input fields with descriptions

# 1. Gender
st.write("**Gender**: Select 'Male' or 'Female' for the applicant's gender.")
gender = st.selectbox("Gender", options=["Male", "Female"])

# 2. Married
st.write("**Married**: Select 'Yes' if the applicant is married, otherwise select 'No'.")
married = st.selectbox("Married", options=["Yes", "No"])

# 3. Dependents
st.write("**Dependents**: Enter the number of dependents the applicant has. Choose '3+' if there are three or more.")
dependents = st.selectbox("Dependents", options=["0", "1", "2", "3+"])

# 4. Education
st.write("**Education**: Select 'Graduate' if the applicant is a graduate, otherwise select 'Not Graduate'.")
education = st.selectbox("Education", options=["Graduate", "Not Graduate"])

# 5. Self-Employed
st.write("**Self-Employed**: Select 'Yes' if the applicant is self-employed, otherwise select 'No'.")
self_employed = st.selectbox("Self-Employed", options=["Yes", "No"])

# 6. Applicant Income
st.write("**Applicant Income**: Enter the applicant's monthly income.")
applicant_income = st.number_input("Applicant Income", min_value=0)

# 7. Coapplicant Income
st.write("**Coapplicant Income**: Enter the coapplicant's monthly income. Enter 0 if there is no coapplicant.")
coapplicant_income = st.number_input("Coapplicant Income", min_value=0)

# 8. Loan Amount
st.write("**Loan Amount**: Enter the loan amount requested by the applicant (in thousands).")
loan_amount = st.number_input("Loan Amount (in thousands)", min_value=0)

# 9. Loan Amount Term
st.write("**Loan Amount Term**: Enter the loan repayment term in months.")
loan_amount_term = st.number_input("Loan Amount Term (in months)", min_value=0)

# 10. Credit History
st.write("**Credit History**: Select 1 if the applicant has a good credit history, otherwise select 0.")
credit_history = st.selectbox("Credit History", options=[1, 0])

# 11. Property Area
st.write("**Property Area**: Select the area type where the applicant's property is located.")
property_area = st.selectbox("Property Area", options=["Urban", "Semiurban", "Rural"])

# Convert inputs to numerical values as per model encoding
gender = 1 if gender == "Male" else 0
married = 1 if married == "Yes" else 0
dependents = 3 if dependents == "3+" else int(dependents)
education = 1 if education == "Graduate" else 0
self_employed = 1 if self_employed == "Yes" else 0
property_area = {"Urban": 2, "Semiurban": 1, "Rural": 0}[property_area]

# Prepare input data for prediction
input_data = np.array([[gender, married, dependents, education, self_employed,
                        applicant_income, coapplicant_income, loan_amount,
                        loan_amount_term, credit_history, property_area]])

# Predict and display results for all three models
if st.button("Predict Loan Approval"):
    logistic_prediction = logistic_model.predict(input_data)
    random_forest_prediction = random_forest_model.predict(input_data)
    svm_prediction = svm_model.predict(input_data)

    # Display predictions for each model
    st.write("### Model Predictions:")
    st.write(f"**Logistic Regression Prediction**: {'Approved' if logistic_prediction[0] == 1 else 'Not Approved'}")
    st.write(f"**Random Forest Prediction**: {'Approved' if random_forest_prediction[0] == 1 else 'Not Approved'}")
    st.write(f"**SVM Prediction**: {'Approved' if svm_prediction[0] == 1 else 'Not Approved'}")
