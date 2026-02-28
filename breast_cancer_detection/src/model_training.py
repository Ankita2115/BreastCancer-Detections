import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

class ModelTrainer:
    """Train and evaluate machine learning models for breast cancer detection."""

    def __init__(self):
        self.models = {
            'logistic_regression': LogisticRegression(max_iter=1000, random_state=42),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingClassifier(random_state=42),
            'svm': SVC(kernel='rbf', probability=True, random_state=42)
        }
        self.trained_models = {}
        self.evaluation_results = {}

    def train_model(self, model_name, X_train, y_train):
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Available: {list(self.models.keys())}")
        model = self.models[model_name]
        model.fit(X_train, y_train)
        self.trained_models[model_name] = model
        print(f"✓ {model_name} trained successfully")
        return model

    def train_all_models(self, X_train, y_train):
        for model_name in self.models.keys():
            self.train_model(model_name, X_train, y_train)

    def evaluate_model(self, model_name, X_test, y_test):
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not trained yet")
        model = self.trained_models[model_name]
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
        self.evaluation_results[model_name] = metrics
        return metrics

    def evaluate_all_models(self, X_test, y_test):
        return {m: self.evaluate_model(m, X_test, y_test) for m in self.trained_models.keys()}

    def get_best_model(self):
        if not self.evaluation_results:
            raise ValueError("No models evaluated yet")
        best_model = max(self.evaluation_results.items(), key=lambda x: x[1]['f1_score'])
        return best_model[0], best_model[1]

    def save_model(self, model_name, filepath):
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.trained_models[model_name], filepath)
        print(f"✓ Model saved to {filepath}")

    def load_model(self, model_name, filepath):
        model = joblib.load(filepath)
        self.trained_models[model_name] = model
        print(f"✓ Model loaded from {filepath}")
        return model

    def predict(self, model_name, X):
        return self.trained_models[model_name].predict(X)

    def predict_proba(self, model_name, X):
        return self.trained_models[model_name].predict_proba(X)