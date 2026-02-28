# Breast Cancer Detection - Machine Learning Project

A comprehensive machine learning project for detecting breast cancer using the Wisconsin Breast Cancer Dataset. This project implements multiple classification models and provides tools for training, evaluation, and prediction.

## Project Overview

This project aims to build and compare different machine learning models to accurately detect breast cancer from diagnostic measurements. The models can help identify malignant tumors with high accuracy.

## Dataset

**Wisconsin Breast Cancer Dataset**
- **Samples**: 569
- **Features**: 30 (computed from digitized images of fine needle aspirate)
- **Target Classes**: 2 (Malignant: 0, Benign: 1)
- **Features include**: radius, texture, perimeter, area, smoothness, compactness, concavity, symmetry, fractal dimension, etc.

## Models Implemented

1. **Logistic Regression** - Linear baseline model
2. **Random Forest** - Ensemble tree-based model
3. **Gradient Boosting** - Advanced ensemble method
4. **Support Vector Machine (SVM)** - Non-linear classifier

## Project Structure

```
breast_cancer_detection/
├── data/                      # Dataset and data files
├── models/                    # Trained model files
├── notebooks/                 # Jupyter notebooks for exploration
├── src/                       # Source code
│   ├── data_preprocessing.py  # Data loading and preprocessing
│   ├── model_training.py      # Model training and evaluation
│   └── main.py               # Main execution script
├── requirements.txt           # Project dependencies
└── README.md                 # This file
```

## Installation

1. Clone or download this project
2. Navigate to the project directory
3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Running the Complete Pipeline

Execute the main script to train all models and evaluate them:

```bash
cd src
python main.py
```

This will:
1. Load the Wisconsin Breast Cancer dataset
2. Preprocess and split the data (80-20 train-test split)
3. Train all four models
4. Evaluate each model
5. Display performance metrics
6. Save the best-performing model

### Using Individual Components

#### Data Preprocessing

```python
from data_preprocessing import load_data, preprocess_data

X, y = load_data()
X_train, X_test, y_train, y_test, scaler = preprocess_data(X, y)
```

#### Model Training and Evaluation

```python
from model_training import ModelTrainer

trainer = ModelTrainer()
trainer.train_model('random_forest', X_train, y_train)
metrics = trainer.evaluate_model('random_forest', X_test, y_test)
predictions = trainer.predict('random_forest', X_test)
```

## Model Evaluation Metrics

Each model is evaluated using:

- **Accuracy**: Proportion of correct predictions
- **Precision**: True positives / (true positives + false positives)
- **Recall**: True positives / (true positives + false negatives)
- **F1 Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the ROC curve

## Key Features

✓ **Multiple Models**: Compare 4 different algorithms
✓ **Comprehensive Evaluation**: 5 different metrics
✓ **Data Preprocessing**: Automatic feature scaling
✓ **Model Persistence**: Save and load trained models
✓ **Easy Prediction**: Simple API for making predictions
✓ **Reproducible**: Fixed random seeds for consistency

## Performance Expectations

Typical model performance on the Wisconsin dataset:

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | ~0.95 | ~0.96 | ~0.96 | ~0.96 | ~0.99 |
| Random Forest | ~0.97 | ~0.97 | ~0.98 | ~0.98 | ~0.99 |
| Gradient Boosting | ~0.97 | ~0.97 | ~0.98 | ~0.98 | ~0.99 |
| SVM | ~0.97 | ~0.96 | ~0.98 | ~0.97 | ~0.99 |

*Note: Actual performance may vary based on random seed and specific data splits.*

## Making Predictions

### Single Sample Prediction

```python
from data_preprocessing import load_data, preprocess_data
from model_training import ModelTrainer

# Load and prepare data
X, y = load_data()
X_train, X_test, y_train, y_test, scaler = preprocess_data(X, y)

# Train model
trainer = ModelTrainer()
trainer.train_model('random_forest', X_train, y_train)

# Make prediction
sample = X_test.iloc[:1]
prediction = trainer.predict('random_forest', sample)
probability = trainer.predict_proba('random_forest', sample)

print(f"Prediction: {'Benign' if prediction[0] == 1 else 'Malignant'}")
print(f"Confidence: {max(probability[0]) * 100:.2f}%")
```

## Future Enhancements

- Hyperparameter tuning using GridSearchCV
- Feature importance analysis
- Cross-validation implementation
- Neural network models (Deep Learning)
- Web interface for predictions
- Real-time prediction API
- Model interpretability with SHAP values

## Requirements

- Python 3.8+
- numpy
- pandas
- scikit-learn
- matplotlib
- seaborn
- jupyter

## License

This project is open source and available for educational and research purposes.

## References

- Wisconsin Breast Cancer Dataset: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html
- Scikit-Learn Documentation: https://scikit-learn.org
- Machine Learning Best Practices: https://www.coursera.org/learn/machine-learning

## Author Notes

This project demonstrates best practices in machine learning including:
- Proper train-test splitting with stratification
- Feature scaling and standardization
- Model comparison and evaluation
- Code organization and modularity
- Documentation and reproducibility

---

**Disclaimer**: This project is for educational purposes only. It should not be used as a substitute for professional medical diagnosis. Always consult with qualified medical professionals for actual cancer detection and treatment.
