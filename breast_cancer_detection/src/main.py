"""
Main script for breast cancer detection using machine learning.
"""

import pandas as pd
from src.data_preprocessing import load_data, preprocess_data
from src.model_training import ModelTrainer


def main():
    """Main execution function."""
    
    print("=" * 60)
    print("Breast Cancer Detection - Machine Learning Project")
    print("=" * 60)
    
    # Load data
    print("\n[1] Loading dataset...")
    X, y = load_data()
    print(f"✓ Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"  Class distribution: {dict(y.value_counts())}")
    
    # Preprocess data
    print("\n[2] Preprocessing data...")
    X_train, X_test, y_train, y_test, scaler = preprocess_data(X, y)
    print(f"✓ Training set: {X_train.shape}")
    print(f"✓ Test set: {X_test.shape}")
    print(f"✓ Features scaled and standardized")
    
    # Train models
    print("\n[3] Training models...")
    trainer = ModelTrainer()
    trainer.train_all_models(X_train, y_train)
    
    # Evaluate models
    print("\n[4] Evaluating models...")
    results = trainer.evaluate_all_models(X_test, y_test)
    
    # Display results
    print("\n" + "=" * 60)
    print("Model Performance Comparison")
    print("=" * 60)
    
    results_df = pd.DataFrame(results).T
    results_df = results_df.round(4)
    print(results_df)
    
    # Get best model
    best_model, best_metrics = trainer.get_best_model()
    print(f"\n✓ Best Model: {best_model}")
    print(f"  F1 Score: {best_metrics['f1_score']:.4f}")
    print(f"  Accuracy: {best_metrics['accuracy']:.4f}")
    print(f"  ROC-AUC: {best_metrics['roc_auc']:.4f}")
    
    # Save best model
    model_path = f"../models/best_model_{best_model}.pkl"
    trainer.save_model(best_model, model_path)
    
    print("\n" + "=" * 60)
    print("Project completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
