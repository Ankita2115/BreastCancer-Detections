from src.data_preprocessing import load_data, preprocess_data
from src.model_training import ModelTrainer

X, y = load_data()
X_train, X_test, y_train, y_test, scaler = preprocess_data(X, y)

trainer = ModelTrainer()
trainer.train_all_models(X_train, y_train)
results = trainer.evaluate_all_models(X_test, y_test)
print(results)
