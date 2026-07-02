import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

X_train = pd.read_csv('X_train.csv').values
X_test = pd.read_csv('X_test.csv').values
y_train = pd.read_csv('y_train.csv').values.ravel()
y_test = pd.read_csv('y_test.csv').values.ravel()

models = {
    'XGBoost': XGBClassifier(
        n_estimators=100, learning_rate=0.1, max_depth=3,
        reg_lambda=1.0, subsample=0.8, colsample_bytree=0.8,
        random_state=42, eval_metric='logloss'
    ),
    'LightGBM': LGBMClassifier(
        n_estimators=100, learning_rate=0.1, max_depth=3,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, verbose=-1
    ),
    'Random Forest': Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('clf', RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42))
    ]),
    'SVM': Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('clf', SVC(kernel='rbf', probability=True, random_state=42))
    ]),
}

for name, model in models.items():
    model.fit(X_train, y_train)
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))
    print(f"{name}: Train={train_acc:.3f}, Test={test_acc:.3f}")
    fname = name.lower().replace(' ', '_') + '_model.pkl'
    joblib.dump(model, fname)
    print(f"  Saved as {fname}")

print("\nAll models trained and saved.")
