// Load dataset information
function loadDatasetInfo() {
    fetch('/api/dataset-info')
        .then(response => response.json())
        .then(data => {
            document.getElementById('totalSamples').textContent = data.samples;
            document.getElementById('totalFeatures').textContent = data.features;
            document.getElementById('benignCases').textContent = data.class_distribution.benign;
            document.getElementById('malignantCases').textContent = data.class_distribution.malignant;
        })
        .catch(error => console.error('Error loading dataset info:', error));
}

// Load model results
function loadModelResults() {
    fetch('/api/results')
        .then(response => response.json())
        .then(data => {
            const tbody = document.getElementById('resultsBody');
            tbody.innerHTML = '';
            
            for (const [model, metrics] of Object.entries(data)) {
                const row = `
                    <tr>
                        <td><strong>${formatModelName(model)}</strong></td>
                        <td>${(metrics.accuracy * 100).toFixed(2)}%</td>
                        <td>${(metrics.precision * 100).toFixed(2)}%</td>
                        <td>${(metrics.recall * 100).toFixed(2)}%</td>
                        <td>${(metrics.f1_score * 100).toFixed(2)}%</td>
                        <td>${(metrics.roc_auc * 100).toFixed(2)}%</td>
                    </tr>
                `;
                tbody.innerHTML += row;
            }
        })
        .catch(error => console.error('Error loading results:', error));
}

// Load best model
function loadBestModel() {
    fetch('/api/best-model')
        .then(response => response.json())
        .then(data => {
            const container = document.getElementById('bestModelInfo');
            const metrics = data.metrics;
            const html = `
                <p style="font-size: 1.2em; margin-bottom: 15px;">
                    <strong>${formatModelName(data.model)}</strong>
                </p>
                <div class="metric">
                    <span class="metric-label">Accuracy:</span>
                    <span class="metric-value">${(metrics.accuracy * 100).toFixed(2)}%</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Precision:</span>
                    <span class="metric-value">${(metrics.precision * 100).toFixed(2)}%</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Recall:</span>
                    <span class="metric-value">${(metrics.recall * 100).toFixed(2)}%</span>
                </div>
                <div class="metric">
                    <span class="metric-label">F1-Score:</span>
                    <span class="metric-value">${(metrics.f1_score * 100).toFixed(2)}%</span>
                </div>
                <div class="metric">
                    <span class="metric-label">ROC-AUC:</span>
                    <span class="metric-value">${(metrics.roc_auc * 100).toFixed(2)}%</span>
                </div>
            `;
            container.innerHTML = html;
        })
        .catch(error => console.error('Error loading best model:', error));
}

// Load available models
function loadModels() {
    fetch('/api/models')
        .then(response => response.json())
        .then(data => {
            const select = document.getElementById('modelSelect');
            select.innerHTML = '<option value="">-- Select a Model --</option>';
            
            data.models.forEach(model => {
                const option = document.createElement('option');
                option.value = model;
                option.textContent = formatModelName(model);
                select.appendChild(option);
            });
        })
        .catch(error => console.error('Error loading models:', error));
}

// Format model name for display
function formatModelName(name) {
    return name
        .split('_')
        .map(word => word.charAt(0).toUpperCase() + word.slice(1))
        .join(' ');
}

// Load sample data
function loadSampleData(type) {
    const samples = {
        benign: [
            13.54, 14.36, 87.46, 566.3, 0.09779, 0.08129, 0.06664, 0.04781, 0.1885, 0.05766,
            0.2699, 0.4085, 1.95, 23.94, 0.00398, 0.01038, 0.01282, 0.00666, 0.01887, 0.01636,
            0.1236, 0.1737, 1.522, 24.64, 0.005667, 0.009613, 0.01265, 0.007938, 0.02183, 0.0182
        ],
        malignant: [
            18.02, 21.26, 122.1, 1001.0, 0.1184, 0.2776, 0.3784, 0.1320, 0.2783, 0.06294,
            0.4601, 1.095, 0.9053, 8.589, 0.005891, 0.04893, 0.04677, 0.01885, 0.01307, 0.001467,
            0.2950, 0.7523, 1.300, 12.46, 0.1212, 0.1922, 0.1999, 0.07975, 0.2871, 0.06859
        ]
    };
    
    const data = samples[type];
    document.getElementById('sampleData').value = data.join(', ');
    document.getElementById('sampleDisplay').textContent = 
        `Sample ${type.charAt(0).toUpperCase() + type.slice(1)} Case:\n[${data.join(', ')}]`;
}

// Make prediction
function makePrediction() {
    const model = document.getElementById('modelSelect').value;
    const featuresText = document.getElementById('sampleData').value;
    
    if (!model) {
        alert('Please select a model');
        return;
    }
    
    if (!featuresText) {
        alert('Please enter feature values');
        return;
    }
    
    try {
        const features = featuresText.split(',').map(f => parseFloat(f.trim()));
        
        if (features.length !== 30) {
            alert(`Please provide exactly 30 features (provided: ${features.length})`);
            return;
        }
        
        if (features.some(isNaN)) {
            alert('All values must be numbers');
            return;
        }
        
        fetch('/api/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                model: model,
                features: features
            })
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                showError('Error: ' + data.error);
                return;
            }
            
            displayPredictionResult(data);
        })
        .catch(error => {
            console.error('Error:', error);
            showError('Failed to make prediction');
        });
    } catch (error) {
        alert('Error parsing features: ' + error.message);
    }
}

// Display prediction result
function displayPredictionResult(data) {
    const resultDiv = document.getElementById('predictionResult');
    const badge = document.getElementById('predictionBadge');
    
    const isBenign = data.prediction === 1;
    const badgeEmoji = isBenign ? '✅' : '⚠️';
    const badgeColor = isBenign ? '#4caf50' : '#f44336';
    
    badge.textContent = badgeEmoji;
    badge.style.color = badgeColor;
    
    document.getElementById('resultPrediction').textContent = data.prediction_label;
    document.getElementById('resultConfidence').textContent = (data.confidence * 100).toFixed(2) + '%';
    document.getElementById('resultMalignant').textContent = (data.probability_malignant * 100).toFixed(2) + '%';
    document.getElementById('resultBenign').textContent = (data.probability_benign * 100).toFixed(2) + '%';
    
    resultDiv.style.display = 'block';
    resultDiv.scrollIntoView({ behavior: 'smooth' });
}

// Show error message
function showError(message) {
    alert(message);
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    loadDatasetInfo();
    loadModelResults();
    loadBestModel();
    loadModels();
    loadSampleData('benign');
});
