# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score, classification_report, auc, confusion_matrix, roc_auc_score, f1_score, precision_score, recall_score
from sklearn.metrics import roc_curve
import xgboost as xgb
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns


# Load your dataset
df = pd.read_csv('balanced_dataset.csv')

# Define your target variable
target = 'Target'

# Define the predictors explicitly
predictors = ['age', 'gender', 'education', 'country', 'nscore', 'escore', 'oscore', 'ascore', 'cscore', 'impulsive', 'ss']

# Split the data into features (X) and target (y)
X = df[predictors]
y = df[target]

# Standardize the data (if needed)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# First split: Separate 5% as the hold-out set
X_temp, X_holdout, y_temp, y_holdout = train_test_split(X_scaled, y, test_size=0.05, random_state=42)

# Second split: Split remaining 95% into train (80%), validation (10%), and test (10%)
X_train, X_temp, y_train, y_temp = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Output dataset sizes
print(f"Train set size: {X_train.shape[0]}")
print(f"Validation set size: {X_val.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")
print(f"Hold-out set size: {X_holdout.shape[0]}")


def evaluate_model(model, X_val, y_val, model_name):
    # Predictions
    y_val_pred = model.predict(X_val)
    y_val_proba = model.predict_proba(X_val)[:, 1] if hasattr(model, "predict_proba") else None

    # Confusion Matrix
    cm = confusion_matrix(y_val, y_val_pred)

    # Print confusion matrix
    print(f"\nConfusion Matrix for {model_name}:")
    print(cm)

    # Classification Report
    print(f"\nClassification Report for {model_name}:")
    print(classification_report(y_val, y_val_pred))

    # Precision, Recall, and F1 Score
    precision = precision_score(y_val, y_val_pred)
    recall = recall_score(y_val, y_val_pred)
    f1 = f1_score(y_val, y_val_pred)

    print(f"Precision for {model_name}: {precision:.4f}")
    print(f"Recall for {model_name}: {recall:.4f}")
    print(f"F1 Score for {model_name}: {f1:.4f}")

    # ROC-AUC Score and Plot
    if y_val_proba is not None and not np.isnan(y_val_proba).any():
        roc_auc = roc_auc_score(y_val, y_val_proba)
        print(f"ROC-AUC for {model_name}: {roc_auc:.4f}")

        # Compute ROC curve
        fpr, tpr, thresholds = roc_curve(y_val, y_val_proba)

        # Plot ROC curve
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve for {model_name}')
        plt.legend(loc='lower right')
        plt.grid(alpha=0.5)
        plt.tight_layout()
        plt.show()
    else:
        print(f"ROC-AUC for {model_name}: Not available (model lacks `predict_proba` or invalid probabilities).")

    # Visualize the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"])
    plt.title(f"Confusion Matrix for {model_name}")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.tight_layout()
    plt.show()


# Stratified K-Fold Cross Validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Metrics to evaluate during cross-validation
scorers = {
    "precision": make_scorer(precision_score),
    "recall": make_scorer(recall_score),
    "f1": make_scorer(f1_score),
    "roc_auc": make_scorer(roc_auc_score, needs_proba=True),
}

precision_scores, recall_scores, f1_scores, roc_auc_scores = [], [], [], []

# Convert y_train to NumPy array
y_train_np = np.array(y_train)

# Perform cross-validation
for train_index, test_index in cv.split(X_train, y_train_np):  # Use y_train_np here
    X_cv_train, X_cv_val = X_train[train_index], X_train[test_index]
    y_cv_train, y_cv_val = y_train_np[train_index], y_train_np[test_index]

    # Train Logistic Regression with Lasso
    log_reg_lasso = LogisticRegression(penalty='l1', solver='liblinear', C=0.1, random_state=42)
    log_reg_lasso.fit(X_cv_train, y_cv_train)

    # Predictions
    y_cv_pred = log_reg_lasso.predict(X_cv_val)
    y_cv_proba = log_reg_lasso.predict_proba(X_cv_val)[:, 1]

    # Metrics
    precision_scores.append(precision_score(y_cv_val, y_cv_pred))
    recall_scores.append(recall_score(y_cv_val, y_cv_pred))
    f1_scores.append(f1_score(y_cv_val, y_cv_pred))
    roc_auc_scores.append(roc_auc_score(y_cv_val, y_cv_proba))

# Calculate mean scores
mean_precision = np.mean(precision_scores)
mean_recall = np.mean(recall_scores)
mean_f1 = np.mean(f1_scores)
mean_roc_auc = np.mean(roc_auc_scores)

print("\nCross-Validation Metrics for Logistic Regression with Lasso:")
print(f"Mean Precision: {mean_precision:.4f}")
print(f"Mean Recall: {mean_recall:.4f}")
print(f"Mean F1 Score: {mean_f1:.4f}")
print(f"Mean ROC-AUC: {mean_roc_auc:.4f}")

# Train on full training set for final evaluation
log_reg_lasso.fit(X_train, y_train)

# Identify selected predictors
coefficients = log_reg_lasso.coef_[0]  # Coefficients for the features
selected_features = [predictors[i] for i in range(len(coefficients)) if coefficients[i] != 0]
excluded_features = [predictors[i] for i in range(len(coefficients)) if coefficients[i] == 0]

print(f"\nSelected Predictors: {selected_features}")
print(f"Excluded Predictors: {excluded_features}")

# Evaluate Logistic Regression on validation set
evaluate_model(log_reg_lasso, X_val, y_val, "Logistic Regression with Lasso")


# 2. Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
evaluate_model(rf, X_val, y_val, "Random Forest")

# # 3. Baseline XGBoost
xgb_baseline = xgb.XGBClassifier(eval_metric='logloss', random_state=42)
xgb_baseline.fit(X_train, y_train)
evaluate_model(xgb_baseline, X_val, y_val, "Baseline XGBoost")

# 4. Hyperparameter Tuning for Random Forest
param_grid_rf = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
}

grid_search_rf = GridSearchCV(estimator=RandomForestClassifier(random_state=42),
                              param_grid=param_grid_rf,
                              scoring='f1',
                              cv=5,
                              verbose=1,
                              n_jobs=-1)

grid_search_rf.fit(X_train, y_train)
print(f"Best Parameters for Random Forest: {grid_search_rf.best_params_}")
best_rf = grid_search_rf.best_estimator_

# Evaluate the optimized Random Forest
evaluate_model(best_rf, X_val, y_val, "Optimized Random Forest")

# 5. Hyperparameter Tuning for XGBoost
param_grid_xgb = {
    "n_estimators": [50, 100, 200],
    "max_depth": [3, 5, 10],
    "learning_rate": [0.01, 0.1, 0.2],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
}

# # Convert X_train and y_train to NumPy arrays
X_train_np = np.array(X_train)
y_train_np = np.array(y_train)

# Manual grid search with StratifiedKFold
best_params = None
best_score = -np.inf

for n_estimators in param_grid_xgb["n_estimators"]:
    for max_depth in param_grid_xgb["max_depth"]:
        for learning_rate in param_grid_xgb["learning_rate"]:
            for subsample in param_grid_xgb["subsample"]:
                for colsample_bytree in param_grid_xgb["colsample_bytree"]:
                    params = {
                        "n_estimators": n_estimators,
                        "max_depth": max_depth,
                        "learning_rate": learning_rate,
                        "subsample": subsample,
                        "colsample_bytree": colsample_bytree,
                    }
                    xgb_model = xgb.XGBClassifier(**params, eval_metric='logloss', random_state=42)

                    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                    f1_scores = []

                    for train_index, test_index in skf.split(X_train_np, y_train_np):
                        # Use NumPy arrays instead of Pandas Series
                        X_cv_train, X_cv_test = X_train_np[train_index], X_train_np[test_index]
                        y_cv_train, y_cv_test = y_train_np[train_index], y_train_np[test_index]

                        xgb_model.fit(X_cv_train, y_cv_train)
                        y_cv_pred = xgb_model.predict(X_cv_test)
                        f1_scores.append(f1_score(y_cv_test, y_cv_pred))

                    mean_f1 = np.mean(f1_scores)

                    if mean_f1 > best_score:
                        best_score = mean_f1
                        best_params = params

print(f"Best Parameters for XGBoost: {best_params}")
best_xgb = xgb.XGBClassifier(**best_params, eval_metric='logloss', random_state=42)
best_xgb.fit(X_train_np, y_train_np)

# Evaluate the optimized XGBoost
evaluate_model(best_xgb, X_val, y_val, "Optimized XGBoost")

print(f"Best Parameters for XGBoost: {best_params}")
best_xgb = xgb.XGBClassifier(**best_params, eval_metric='logloss', random_state=42)
best_xgb.fit(X_train, y_train)

# Evaluate the optimized XGBoost
evaluate_model(best_xgb, X_val, y_val, "Optimized XGBoost")

# 6. Evaluate SVM
svm_model = SVC(kernel='rbf', probability=True, random_state=42)
svm_model.fit(X_train, y_train)
evaluate_model(svm_model, X_val, y_val, "SVM")

# 7. Evaluate Decision Tree
dt = DecisionTreeClassifier(max_depth=5, random_state=42)
dt.fit(X_train, y_train)
evaluate_model(dt, X_val, y_val, "Decision Tree")

# 8. Evaluate Bagging Classifier
bagging_clf = BaggingClassifier(estimator=DecisionTreeClassifier(max_depth=5, random_state=42), n_estimators=50, random_state=42)
bagging_clf.fit(X_train, y_train)
evaluate_model(bagging_clf, X_val, y_val, "Bagging Classifier")

# 9. Final Evaluation on Hold-Out Set
models = {
    "Logistic Regression": log_reg_lasso,
    "Random Forest": rf,
    "Optimized Random Forest": best_rf,
    "Baseline XGBoost": xgb_baseline,
    "Optimized XGBoost": best_xgb,
    "SVM": svm_model,
    "Decision Tree": dt,
    "Bagging Classifier": bagging_clf
}

for model_name, model in models.items():
    y_holdout_pred = model.predict(X_holdout)
    print(f"Hold-out - {model_name} Accuracy: {accuracy_score(y_holdout, y_holdout_pred):.4f}")
    print(f"Hold-out - {model_name} Precision: {precision_score(y_holdout, y_holdout_pred):.4f}")
    print(f"Hold-out - {model_name} Recall: {recall_score(y_holdout, y_holdout_pred):.4f}")
    print(f"Hold-out - {model_name} F1 Score: {f1_score(y_holdout, y_holdout_pred):.4f}")


# Function to plot combined ROC curve
def plot_combined_roc(models, X_val, y_val):
    plt.figure(figsize=(10, 8))
    
    for model_name, model in models.items():
        # Check if the model supports `predict_proba`
        if hasattr(model, "predict_proba"):
            y_val_proba = model.predict_proba(X_val)[:, 1]
        elif hasattr(model, "decision_function"):
            y_val_proba = model.decision_function(X_val)
        else:
            print(f"Skipping {model_name}: no probabilistic output available.")
            continue
        
        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(y_val, y_val_proba)
        auc = roc_auc_score(y_val, y_val_proba)
        
        # Plot ROC curve
        plt.plot(fpr, tpr, label=f"{model_name} (AUC = {auc:.2f})")
    
    # Plot diagonal line
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--", label="Random Guessing")
    
    # Labels, title, and legend
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison Across Models")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.5)
    plt.tight_layout()
    plt.show()

# Prepare models for evaluation
models = {
    "Logistic Regression": log_reg_lasso,
    "Random Forest": rf,
    "Optimized Random Forest": best_rf,
    "Baseline XGBoost": xgb_baseline,  # Added Baseline XGBoost
    "Optimized XGBoost": best_xgb,
    "SVM": svm_model,
    "Decision Tree": dt,
    "Bagging Classifier": bagging_clf
}

# Generate the combined ROC curve
plot_combined_roc(models, X_val, y_val)


# Function to evaluate a model and return metrics
def get_model_metrics(model, X_data, y_data, model_name):
    metrics = {"Model": model_name}
    
    # Predictions
    y_pred = model.predict(X_data)
    y_proba = model.predict_proba(X_data)[:, 1] if hasattr(model, "predict_proba") else None

    # Metrics
    metrics["Accuracy"] = accuracy_score(y_data, y_pred)
    metrics["Precision"] = precision_score(y_data, y_pred)
    metrics["Recall"] = recall_score(y_data, y_pred)
    metrics["F1 Score"] = f1_score(y_data, y_pred)
    metrics["AUC"] = roc_auc_score(y_data, y_proba) if y_proba is not None else "N/A"

    return metrics

# Collect metrics for Validation Set
validation_results = []
for model_name, model in models.items():
    metrics = get_model_metrics(model, X_val, y_val, model_name)
    validation_results.append(metrics)

# Collect metrics for Hold-Out Set
holdout_results = []
for model_name, model in models.items():
    metrics = get_model_metrics(model, X_holdout, y_holdout, model_name)
    holdout_results.append(metrics)

# Combine results into a single DataFrame
validation_df = pd.DataFrame(validation_results)
holdout_df = pd.DataFrame(holdout_results)

# Combine the two tables into a single DataFrame with a column indicating the dataset
validation_df["Dataset"] = "Validation"
holdout_df["Dataset"] = "Hold-Out"
combined_results = pd.concat([validation_df, holdout_df], ignore_index=True)

# Save the combined results to an HTML file
combined_html_table = combined_results.to_html(index=False, border=1, justify="center")
with open("model_comparison_combined.html", "w") as f:
    f.write(combined_html_table)

print("Comparison table saved as 'model_comparison_combined.html'")
