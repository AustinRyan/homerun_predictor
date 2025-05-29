# backend/train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
import joblib
import os

# Ensure the models directory exists
if not os.path.exists('models'):
    os.makedirs('models')

def train_hr_model(features_csv: str = "data/features.csv", model_path: str = "models/hr_model.pkl"):
    """Loads features, trains a Logistic Regression model, evaluates, and saves it.

    Args:
        features_csv: Path to the CSV file containing features and target.
        model_path: Path to save the trained model.
    """
    print(f"Loading data from {features_csv}...")
    try:
        df = pd.read_csv(features_csv)
    except FileNotFoundError:
        print(f"Error: Features file {features_csv} not found. Please run prepare_features.py first.")
        return

    if df.empty:
        print("The feature DataFrame is empty. Aborting training.")
        return

    # Define features (X) and target (y)
    # Ensure 'is_hr' is not accidentally included in X if it was in features.csv for other reasons
    feature_columns = [
        'launch_speed',
        'launch_angle',
        'ideal_launch_angle',
        'hard_hit',
        'estimated_woba_using_speedangle',
        'hit_distance_sc'
    ]
    X = df[feature_columns]
    y = df['is_hr']

    print(f"Feature columns being used for training: {X.columns.tolist()}")
    print(f"Target distribution:\n{y.value_counts(normalize=True)}")

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Training set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")

    # Initialize and train the Logistic Regression model
    print("Training Logistic Regression model...")
    # Increased max_iter for convergence, class_weight='balanced' for imbalanced target
    model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced') 
    model.fit(X_train, y_train)

    # Evaluate the model on the test set
    print("\nEvaluating model on the test set...")
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"AUC: {auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Print feature coefficients
    print("\nModel Coefficients:")
    coefficients = pd.DataFrame({'feature': X.columns, 'coefficient': model.coef_[0]})
    print(coefficients.sort_values(by='coefficient', ascending=False))

    # Save the trained model
    print(f"\nSaving trained model to {model_path}...")
    joblib.dump(model, model_path)
    print(f"Model saved successfully to {model_path}")
    print("Model training and evaluation process finished.")

if __name__ == "__main__":
    train_hr_model()
