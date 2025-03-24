# Drug_Consumption_Prediction
 Using personality traits and demographics this repository try to predict the risk of drug consumption

## Instructions to run the py files:
- 1. Is advised to create a virtual environment and install the modules:
- - #pip install ucimlrepo - v. 0.0.7
- - #pip install pandas - v. 2.2.3
- - #pip install numpy - v. 2.2.1
- - #pip install matplotlib - v. 3.10.0
- - #pip install seaborn - v. 0.13.2
- - #pip install scikit-learn - v. 1.6.1
- - #pip install imbalanced-learn - v. 0.13.0
- - #pip install xgboost - v. 2.1.3
- - #pip install Pweave - v. 0.30.3

- - python version 3.12

- 2. Run the first file '1_preprocessing.py' to generate file: 'balanced_dataset.csv'
- 3. Use the 'balanced_dataset.csv' as source of data to run the file '2_models'

## Content of py files:

- 1_preprocessing.py: This file contain steps to preprocess the data used in the machine learning models, the preprocessing steps was:
- - exploratory data analysis
- - descriptive analysis
- - target variables creation
- - oversampling technique
- - saving the balanced_dataset.csv file

- 2_models.py: This file contain the machine learning models applied to the dataset, such as:
- - Logistic Regression with L1 regularization and cross validation
- - Random Forest (baseline)
- - Random Forest (Optimized with hyperparameters with cross validation)
- - XGBoost (baseline)
- - XGBoost (Optimized with hyperparameters with cross validation)
- - SVM 
- - Decision Tree
- - Bagging Classifier

This is a group work designed by students from the Universität Luzern as a semester project for the course Supervised Machine Learning.

Names: Camila Batista Rahal, Elena Maestre Soteras 
Fall semester, 2024
Universität Luzern

