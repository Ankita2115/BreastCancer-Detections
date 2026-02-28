"""
Flask web application for Breast Cancer Detection
"""

from flask import Flask, render_template, request, jsonify  # type: ignore[import]

import pandas as pd  # type: ignore[import]

# imports from src package
from src.data_preprocessing import load_data, preprocess_data
from src.model_training import ModelTrainer

app = Flask(__name__)

# Global variables for models
trainer = None
X_test = None
y_test = None
model_results = None

def initialize_models():
    """Initialize and train models on first run"""
    global trainer, X_test, y_test, model_results
    
    try:
        print("Loading and preparing data...")
        X, y = load_data()
        X_train, X_test, y_train, y_test, scaler = preprocess_data(X, y)
        
        print("Training models...")
        trainer = ModelTrainer()
        trainer.train_all_models(X_train, y_train)
        
        print("Evaluating models...")
        model_results = trainer.evaluate_all_models(X_test, y_test)
        
        return True
    except Exception as e:
        print(f"Error during initialization: {e}")
        return False


@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')


@app.route('/api/models')
def get_models():
    """Get list of available models"""
    if trainer is None:
        return jsonify({'error': 'Models not initialized'}), 500
    
    models = list(trainer.trained_models.keys())
    return jsonify({'models': models})

@app.route('/api/results')
def get_results():
    """Get model evaluation results"""
    if model_results is None:
        return jsonify({'error': 'Models not evaluated'}), 500
    
    # Convert to JSON-serializable format
    results = {}
    for model_name, metrics in model_results.items():
        results[model_name] = {k: float(v) for k, v in metrics.items()}
    
    return jsonify(results)


@app.route('/api/best-model')
def get_best_model():
    """Get the best performing model"""
    if trainer is None or model_results is None:
        return jsonify({'error': 'Models not initialized'}), 500
    
    best_model, best_metrics = trainer.get_best_model()
    metrics = {k: float(v) for k, v in best_metrics.items()}
    
    return jsonify({
        'model': best_model,
        'metrics': metrics
    })


@app.route('/api/predict', methods=['POST'])
def predict():
    """Make predictions with a specific model"""
    try:
        data = request.json
        model_name = data.get('model')
        features = data.get('features')
        
        if not model_name or not features:
            return jsonify({'error': 'Missing model name or features'}), 400
        
        if trainer is None:
            return jsonify({'error': 'Models not initialized'}), 500
        
        # Create DataFrame with proper shape
        X, _ = load_data()
        feature_names = X.columns.tolist()
        
        # Normalize features using same scaler
        from src.data_preprocessing import preprocess_data
        X_dummy, _, _, _, scaler = preprocess_data(X, pd.Series([0] * len(X)))
        
        # Create feature array
        feature_array = [[float(f) for f in features]]
        features_scaled = scaler.transform(feature_array)
        
        # Predict
        prediction = trainer.predict(model_name, features_scaled)[0]
        probability = trainer.predict_proba(model_name, features_scaled)[0]
        
        return jsonify({
            'prediction': int(prediction),
            'prediction_label': 'Benign' if prediction == 1 else 'Malignant',
            'probability_malignant': float(probability[0]),
            'probability_benign': float(probability[1]),
            'confidence': float(max(probability))
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/dataset-info')
def get_dataset_info():
    """Get dataset information"""
    try:
        X, y = load_data()
        
        return jsonify({
            'samples': int(X.shape[0]),
            'features': int(X.shape[1]),
            'feature_names': X.columns.tolist(),
            'class_distribution': {
                'malignant': int((y == 0).sum()),
                'benign': int((y == 1).sum())
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404


if __name__ == '__main__':
    print("Initializing Breast Cancer Detection System...")
    if initialize_models():
        print("✓ Models trained successfully!")
        print("Starting Flask server on http://localhost:5000")
        app.run(debug=True, port=5000)
    else:
        print("✗ Failed to initialize models")
