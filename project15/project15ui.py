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
data=pd.read_csv('project15/titanic_train.csv')

#Using IQR to remove outliers

def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# Apply outlier removal to the selected columns
data_clean = remove_outliers(data, 'Age')
data_clean = remove_outliers(data, 'Fare')
data_clean = remove_outliers(data, 'SibSp')
data_clean = remove_outliers(data, 'Parch')

# Checking the clean data's shape
# data_clean.shape

# Removing unneccessary columns
data_clean.drop(columns=['PassengerId','Name','Ticket','Cabin'],inplace=True)

# Preprocessing
# Handle missing values and encode categorical features

data_clean['Age'].fillna(data_clean['Age'].median(), inplace=True)

# For 'Embarked', we will fill missing values with the most frequent value (mode).
data_clean['Embarked'].fillna(data_clean['Embarked'].mode()[0], inplace=True)

# Checking clean data's null values
# data_clean.isna().sum()

#Encoding for categorical columns

from sklearn.preprocessing import LabelEncoder

# Label Encoding for categorical columns 'Sex' and 'Embarked'
label_encoder = LabelEncoder()

# Apply label encoding to 'Sex' and 'Embarked'
data_clean['Sex'] = label_encoder.fit_transform(data_clean['Sex'])
data_clean['Embarked'] = label_encoder.fit_transform(data_clean['Embarked'])

# """## Test Train Split"""

from sklearn.model_selection import train_test_split

# Defining features (X) and target (y)
X = data_clean.drop(columns=['Survived'])
y = data_clean['Survived']

# Splitting the dataset into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Display the shape of the training and test sets
X_train.shape, X_test.shape, y_train.shape, y_test.shape

# Standardize the features (PCA or model scaling may require this step)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# """## Defining a function that checks the model's Accuracy, MAE, MSE, R2 Score"""

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
from sklearn.metrics import accuracy_score
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

# Decision Tree Classifier
decision_tree = DecisionTreeClassifier()

# Fit the model on the training set
decision_tree.fit(X_train, y_train)

# # Print training metrics
# metrics_score(decision_tree, X_train, X_test, y_train, y_test, train=True)

# # Print test metrics
# metrics_score(decision_tree, X_train, X_test, y_train, y_test, train=False)

# Cross-validation accuracy (5-fold)
cv_scores_tree = cross_val_score(decision_tree, X_train, y_train, cv=5, scoring='accuracy')
# print(f'Decision Tree Cross-Validation Accuracy: {cv_scores_tree.mean():.4f}')

# """## 3rd Model: Random Forest Classifier"""

from sklearn.ensemble import RandomForestClassifier

# Random Forest Classifier
random_forest = RandomForestClassifier()

# Fit the model on the training set
random_forest.fit(X_train, y_train)

# Print training metrics
# metrics_score(random_forest, X_train, X_test, y_train, y_test, train=True)

# # Print test metrics
# metrics_score(random_forest, X_train, X_test, y_train, y_test, train=False)

# Cross-validation accuracy (5-fold)
cv_scores_forest = cross_val_score(random_forest, X_train, y_train, cv=5, scoring='accuracy')
# print(f'Random Forest Cross-Validation Accuracy: {cv_scores_forest.mean():.4f}')

# """## 4th Model: KNN"""

from sklearn.neighbors import KNeighborsClassifier

# K-Nearest Neighbors Classifier
knn = KNeighborsClassifier()

# Fit the model on the training set
knn.fit(X_train, y_train)

# Print training metrics
# metrics_score(knn, X_train, X_test, y_train, y_test, train=True)

# # Print test metrics
# metrics_score(knn, X_train, X_test, y_train, y_test, train=False)

# Cross-validation accuracy (5-fold)
cv_scores_knn = cross_val_score(knn, X_train, y_train, cv=5, scoring='accuracy')
# print(f'KNN Cross-Validation Accuracy: {cv_scores_knn.mean():.4f}')

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

# """## Lets perform Hyperparameter Tuning on the best model, which is Random Forest"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# Define the hyperparameter grid for Random Forest
param_grid = {
    'n_estimators': [100, 200, 300],  # Number of trees in the forest
    'max_depth': [10, 20, 30, None],  # Maximum depth of the trees
    'min_samples_split': [2, 5, 10],  # Minimum number of samples to split a node
    'min_samples_leaf': [1, 2, 4]     # Minimum number of samples required at each leaf node
}

# Initialize the Random Forest Classifier
random_forest = RandomForestClassifier(random_state=42)

# GridSearchCV to tune hyperparameters (set verbose=0 to suppress detailed output)
grid_search_rf = GridSearchCV(random_forest, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=0)

# Fit GridSearchCV
grid_search_rf.fit(X_train, y_train)

# Print the best parameters and the best score
# print(f'Best Parameters: {grid_search_rf.best_params_}')
# print(f'Best Cross-Validation Accuracy: {grid_search_rf.best_score_:.4f}')

# Evaluate the best model on the test set
best_rf_model = grid_search_rf.best_estimator_
test_accuracy = accuracy_score(y_test, best_rf_model.predict(X_test))
# print(f'Test Accuracy of the Best Random Forest Model: {test_accuracy:.4f}')

# """## Saving the model to a file using pickle"""

import pickle
# Save the model to a file using pickle
with open('random_forest_model.pkl', 'wb') as model_file:
    pickle.dump(best_rf_model, model_file)

# Load the model from the file
with open('random_forest_model.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)

# Use the loaded model to make predictions
predictions = loaded_model.predict(X_test)

# """## Testing the Model"""

# Load the saved model
with open('random_forest_model.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)

# Example passenger data (same features as training data)
# [Pclass, Sex (1=male, 0=female), Age, SibSp, Parch, Fare, Embarked (0, 1, or 2)]
test_passenger_data = [[3, 1, 25, 0, 0, 7.25, 2],  # Example 1
            [1, 0, 38, 1, 0, 71.2833, 0]]  # Example 2 (you can add more rows)

# Use the loaded model to predict survival for new examples
predictions = loaded_model.predict(test_passenger_data)

# Output the predictions (1 = survived, 0 = did not survive)
# print("Predictions for new examples:", predictions)

# """### This means the first passenger is predicted to not survive and the second passenger is predicted to survive.
# 1st Passenger: Third-class male, aged 25, no relatives aboard, fare of 7.25, embarked from Southampton\
# 2nd Passenger: First-class female, aged 38, traveling with one sibling/spouse, fare of 71.28, embarked from Cherbourg\
# Our model prediction is therefore correct!
# """

## Streamlit Implementation
# app.py

import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the trained Random Forest model
with open('random_forest_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Streamlit app title
st.title("Titanic Survival Prediction App")
st.write("Enter the passenger details below to predict if they would survive.")

# Input fields for passenger data
pclass = st.selectbox("Passenger Class (Pclass)", options=[1, 2, 3])
sex = st.selectbox("Gender", options=["Male", "Female"])
age = st.slider("Age", min_value=0, max_value=80, value=25)
sibsp = st.number_input("Number of Siblings/Spouses Aboard (SibSp)", min_value=0, max_value=10, value=0)
parch = st.number_input("Number of Parents/Children Aboard (Parch)", min_value=0, max_value=10, value=0)
fare = st.number_input("Fare", min_value=0.0, max_value=500.0, value=30.0)
embarked = st.selectbox("Port of Embarkation (Embarked)", options=["C", "Q", "S"])

# Convert categorical inputs to numerical values for the model
sex = 1 if sex == "Male" else 0
embarked = {"C": 0, "Q": 1, "S": 2}[embarked]

# Prepare the input data for prediction
input_data = np.array([[pclass, sex, age, sibsp, parch, fare, embarked]])

# Predict and display the result
if st.button("Predict Survival"):
    prediction = model.predict(input_data)
    result = "Survive" if prediction[0] == 1 else "not survive"
    st.write(f"Prediction: The passenger would **{result}**.")
