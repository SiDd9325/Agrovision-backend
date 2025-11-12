from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io
import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelEncoder, StandardScaler
from werkzeug.utils import secure_filename
import traceback
from datetime import datetime

app = Flask(__name__)
CORS(app)

# ==================== CONFIGURATION ====================
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Create upload folder
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ==================== PATHS ====================
script_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(script_dir, "best_fusion_crop_final.h5")  # Updated model name
CSV_PATH = os.path.join(script_dir, "DATASET", "tomato_disease_dataset .csv")
SCALER_PATH = os.path.join(script_dir, "scaler_params.json")  # Scaler parameters
LABEL_PATH = os.path.join(script_dir, "label_mapping.json")  # Label mapping

# ==================== GLOBAL VARIABLES ====================
model = None
df = None
scaler = None
disease_le = None
tab_features_model = ['Humidity', 'Temperature', 'Soil_pH']

# ==================== HELPER FUNCTIONS ====================
def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def validate_image(file):
    """
    Validate uploaded image file
    Returns: (is_valid, error_message, image_object)
    """
    if not file:
        return False, "No file provided", None
    
    if file.filename == '':
        return False, "No file selected", None
    
    if not allowed_file(file.filename):
        return False, f"File type not allowed. Allowed types: {', '.join(app.config['ALLOWED_EXTENSIONS'])}", None
    
    try:
        # Read file content
        file_bytes = file.read()
        file.seek(0)  # Reset file pointer
        
        # Check file size
        if len(file_bytes) == 0:
            return False, "Empty file uploaded", None
        
        # Try to open and verify it's a valid image
        image = Image.open(io.BytesIO(file_bytes))
        
        # Check image dimensions
        if image.size[0] < 10 or image.size[1] < 10:
            return False, "Image dimensions too small (minimum 10x10 pixels)", None
        
        if image.size[0] > 5000 or image.size[1] > 5000:
            return False, "Image dimensions too large (maximum 5000x5000 pixels)", None
        
        return True, None, image
    
    except Exception as e:
        return False, f"Invalid image file: {str(e)}", None


def validate_tabular_input(data):
    """
    Validate tabular input data
    Returns: (is_valid, error_message, processed_data)
    """
    try:
        # Check if data is provided
        if not data:
            return False, "No tabular data provided", None
        
        # Check required fields
        required_fields = ['humidity', 'temperature', 'soil_ph']
        missing_fields = [field for field in required_fields if field not in data]
        
        if missing_fields:
            return False, f"Missing required fields: {', '.join(missing_fields)}", None
        
        # Validate data types and ranges
        try:
            humidity = float(data['humidity'])
            temperature = float(data['temperature'])
            soil_ph = float(data['soil_ph'])
        except (ValueError, TypeError):
            return False, "All tabular values must be numeric", None
        
        # Validate ranges
        if not (0 <= humidity <= 100):
            return False, "Humidity must be between 0 and 100", None
        
        if not (-50 <= temperature <= 60):
            return False, "Temperature must be between -50°C and 60°C", None
        
        if not (0 <= soil_ph <= 14):
            return False, "Soil pH must be between 0 and 14", None
        
        processed_data = {
            'Humidity': humidity,
            'Temperature': temperature,
            'Soil_pH': soil_ph
        }
        
        return True, None, processed_data
    
    except Exception as e:
        return False, f"Error validating tabular data: {str(e)}", None


def preprocess_image(image, target_size=(128, 128)):  # Updated to 224x224
    """
    Preprocess image for model prediction
    """
    try:
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize image
        image = image.resize(target_size, Image.Resampling.LANCZOS)
        
        # Convert to array and normalize
        img_array = img_to_array(image) / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    
    except Exception as e:
        raise ValueError(f"Error preprocessing image: {str(e)}")


def load_model_and_data():
    """Load model, CSV data, and prepare encoders"""
    global model, df, scaler, disease_le
    
    try:
        print("🔹 Loading trained fusion model...")
        model = load_model(MODEL_PATH)
        print("✅ Model loaded successfully")
        
        print("🔹 Loading CSV data...")
        df = pd.read_csv(CSV_PATH)
        
        # Standardize column names
        df.rename(columns={
            'Disease Name': 'Disease_Name',
            'Leaf Color': 'Leaf_Color',
            'Spots Present': 'Spots_Present',
            'Humidity (%)': 'Humidity',
            'Temperature (°C)': 'Temperature',
            'Soil pH': 'Soil_pH'
        }, inplace=True)
        
        # Create normalized disease name
        df['Disease_Name_Norm'] = df['Disease_Name'].str.strip().str.lower()
        
        # Drop rows with missing values
        df = df.dropna(subset=tab_features_model + ['Disease_Name'])
        
        # Normalize numeric tabular features
        scaler = StandardScaler()
        scaler.fit(df[tab_features_model])
        
        # Encode target labels
        disease_le = LabelEncoder()
        disease_le.fit(df['Disease_Name'])
        
        print("✅ CSV data loaded and preprocessed")
        print(f"✅ Found {len(disease_le.classes_)} disease classes")
        
        return True
    
    except Exception as e:
        print(f"❌ Error loading model/data: {str(e)}")
        traceback.print_exc()
        return False


# ==================== API ENDPOINTS ====================
@app.route('/', methods=['GET'])
def home():
    """Root endpoint - Welcome message"""
    return jsonify({
        'message': '🍅 Tomato Disease Detection API',
        'status': 'running',
        'version': '1.0',
        'endpoints': {
            'health': {
                'url': '/health',
                'method': 'GET',
                'description': 'Check API health status'
            },
            'info': {
                'url': '/info',
                'method': 'GET',
                'description': 'Get model information'
            },
            'predict': {
                'url': '/predict',
                'method': 'POST',
                'description': 'Predict disease with custom tabular data',
                'parameters': {
                    'image': 'Image file (JPG/PNG)',
                    'humidity': 'Humidity % (0-100)',
                    'temperature': 'Temperature °C (-50 to 60)',
                    'soil_ph': 'Soil pH (0-14)'
                }
            },
            'predict_auto': {
                'url': '/predict-auto',
                'method': 'POST',
                'description': 'Predict disease with automatic tabular data',
                'parameters': {
                    'image': 'Image file (JPG/PNG) - only parameter needed'
                }
            }
        },
        'model_loaded': model is not None,
        'timestamp': datetime.now().isoformat()
    }), 200


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'data_loaded': df is not None,
        'timestamp': datetime.now().isoformat()
    }), 200


@app.route('/info', methods=['GET'])
def model_info():
    """Get model information"""
    if model is None or disease_le is None:
        return jsonify({
            'error': 'Model not loaded'
        }), 500
    
    return jsonify({
        'model_type': 'Fusion Model (Image + Tabular)',
        'image_input_size': '224x224',  # Updated
        'tabular_features': tab_features_model,
        'disease_classes': disease_le.classes_.tolist(),
        'num_classes': len(disease_le.classes_),
        'allowed_image_formats': list(app.config['ALLOWED_EXTENSIONS']),
        'max_file_size_mb': app.config['MAX_CONTENT_LENGTH'] / (1024 * 1024),
        'excluded_from_predictions': ['Tomato Spotted Wilt'],
        'note': 'Check if a Healthy class exists in disease_classes',
        'model_architecture': 'EfficientNetB0 + Tabular Features',
        'improvements': [
            'Increased image size to 224x224',
            'EfficientNetB0 backbone',
            'Better data augmentation',
            'Confidence-based health assessment'
        ]
    }), 200


@app.route('/predict', methods=['POST'])
def predict():
    """
    Main prediction endpoint
    Expects:
    - image: file upload
    - humidity: float (0-100)
    - temperature: float (-50 to 60)
    - soil_ph: float (0-14)
    """
    try:
        # Check if model is loaded
        if model is None:
            return jsonify({
                'error': 'Model not initialized',
                'message': 'Server error: Model failed to load'
            }), 500
        
        # ==================== IMAGE VALIDATION ====================
        if 'image' not in request.files:
            return jsonify({
                'error': 'No image provided',
                'message': 'Please upload an image file'
            }), 400
        
        file = request.files['image']
        is_valid, error_msg, image = validate_image(file)
        
        if not is_valid:
            return jsonify({
                'error': 'Invalid image',
                'message': error_msg
            }), 400
        
        # ==================== TABULAR DATA VALIDATION ====================
        tabular_data = {
            'humidity': request.form.get('humidity'),
            'temperature': request.form.get('temperature'),
            'soil_ph': request.form.get('soil_ph')
        }
        
        is_valid, error_msg, processed_data = validate_tabular_input(tabular_data)
        
        if not is_valid:
            return jsonify({
                'error': 'Invalid tabular data',
                'message': error_msg
            }), 400
        
        # ==================== PREPROCESSING ====================
        # Preprocess image
        img_array = preprocess_image(image)
        
        # Prepare tabular input
        tab_input = np.array([[
            processed_data['Humidity'],
            processed_data['Temperature'],
            processed_data['Soil_pH']
        ]])
        
        # Scale tabular input
        tab_input_scaled = scaler.transform(tab_input)
        
        # ==================== PREDICTION ====================
        pred_probs = model.predict([img_array, tab_input_scaled], verbose=0)
        
        # Get all disease names
        all_diseases = disease_le.inverse_transform(np.arange(len(pred_probs[0])))
        
        # Check if model has healthy class
        healthy_variants = ['healthy', 'Healthy', 'HEALTHY']
        has_healthy_class = any(any(variant in disease for variant in healthy_variants) for disease in all_diseases)
        
        # EXCLUDE "Tomato Spotted Wilt" from predictions (optional)
        excluded_diseases = []  # Empty list - include all diseases now since we have healthy
        # If you still want to exclude: excluded_diseases = ["Tomato Spotted Wilt"]
        
        # Find indices of excluded diseases
        excluded_indices = []
        for i, disease in enumerate(all_diseases):
            if disease in excluded_diseases:
                excluded_indices.append(i)
        
        # Create filtered probabilities
        pred_probs_filtered = pred_probs[0].copy()
        for idx in excluded_indices:
            pred_probs_filtered[idx] = -1
        
        # Get prediction
        pred_class = np.argmax(pred_probs_filtered)
        pred_label = disease_le.inverse_transform([pred_class])[0]
        confidence = float(pred_probs[0][pred_class] * 100)
        
        # Get top 5 predictions (excluding excluded diseases)
        sorted_indices = np.argsort(pred_probs_filtered)[::-1]
        top_predictions = []
        for idx in sorted_indices:
            disease_name = disease_le.inverse_transform([idx])[0]
            if disease_name not in excluded_diseases and len(top_predictions) < 5:
                top_predictions.append({
                    'disease': disease_name,
                    'confidence': float(pred_probs[0][idx] * 100)
                })
        
        # Get top 3 for backward compatibility
        top_3_predictions = top_predictions[:3]
        
        # Get all class probabilities (excluding excluded diseases)
        all_probabilities = {
            disease_le.inverse_transform([i])[0]: float(pred_probs[0][i] * 100)
            for i in range(len(pred_probs[0]))
            if disease_le.inverse_transform([i])[0] not in excluded_diseases
        }
        
        # Add debug info - show all probabilities including excluded ones
        all_probabilities_debug = {
            disease_le.inverse_transform([i])[0]: float(pred_probs[0][i] * 100)
            for i in range(len(pred_probs[0]))
        }
        
        # ==================== HEALTH ASSESSMENT ====================
        # Check if predicted class is healthy
        is_healthy_prediction = any(variant in pred_label.lower() for variant in healthy_variants)
        
        if has_healthy_class:
            # Model has healthy class - use prediction directly
            if is_healthy_prediction:
                health_status = {
                    'status': 'healthy',
                    'message': f'Plant appears healthy (confidence: {confidence:.1f}%)',
                    'recommendation': 'Continue regular monitoring and care'
                }
            else:
                # Disease detected
                if confidence >= 80:
                    health_status = {
                        'status': 'disease_detected',
                        'message': f'Disease detected with high confidence',
                        'recommendation': 'Take immediate treatment action for ' + pred_label
                    }
                elif confidence >= 60:
                    health_status = {
                        'status': 'disease_likely',
                        'message': f'Disease likely present (medium confidence)',
                        'recommendation': 'Monitor closely and consider treatment for ' + pred_label
                    }
                else:
                    health_status = {
                        'status': 'uncertain',
                        'message': f'Low confidence - symptoms unclear',
                        'recommendation': 'Continue monitoring, re-check in 2-3 days'
                    }
        else:
            # Model doesn't have healthy class - use confidence thresholds
            if confidence < 50:
                health_status = {
                    'status': 'possibly_healthy',
                    'message': 'Low confidence in disease prediction. Plant may be healthy or have symptoms not in training data.',
                    'recommendation': 'Manual inspection recommended'
                }
            elif confidence < 70:
                health_status = {
                    'status': 'uncertain',
                    'message': 'Medium confidence. Disease symptoms may be unclear or early stage.',
                    'recommendation': 'Monitor plant and re-check in a few days'
                }
            else:
                health_status = {
                    'status': 'disease_detected',
                    'message': 'High confidence in disease detection.',
                    'recommendation': 'Take appropriate treatment measures'
                }
        
        response = {
            'success': True,
            'prediction': {
                'disease': pred_label,
                'confidence': round(confidence, 2),
                'confidence_level': 'high' if confidence >= 80 else 'medium' if confidence >= 60 else 'low'
            },
            'health_assessment': health_status,
            'top_predictions': top_3_predictions,
            'all_predictions': top_predictions,  # Show top 5
            'input_data': {
                'image_size': image.size,
                'humidity': processed_data['Humidity'],
                'temperature': processed_data['Temperature'],
                'soil_ph': processed_data['Soil_pH']
            },
            'all_probabilities': {k: round(v, 2) for k, v in all_probabilities.items()},
            'model_info': {
                'note': 'This model was trained only on diseased plants. It cannot detect healthy plants.',
                'trained_classes': len(disease_le.classes_),
                'has_healthy_class': False
            },
            'debug_info': {
                'all_classes_including_excluded': {k: round(v, 2) for k, v in all_probabilities_debug.items()},
                'excluded_diseases': excluded_diseases,
                'note': 'No healthy class exists in this model'
            },
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response), 200
    
    except Exception as e:
        print(f"❌ Prediction error: {str(e)}")
        traceback.print_exc()
        return jsonify({
            'error': 'Prediction failed',
            'message': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500


@app.route('/predict-auto', methods=['POST'])
def predict_auto():
    """
    Prediction with automatic tabular data (uses average values from CSV)
    Only requires image upload
    """
    try:
        # Check if model is loaded
        if model is None:
            return jsonify({
                'error': 'Model not initialized',
                'message': 'Server error: Model failed to load'
            }), 500
        
        # ==================== IMAGE VALIDATION ====================
        if 'image' not in request.files:
            return jsonify({
                'error': 'No image provided',
                'message': 'Please upload an image file'
            }), 400
        
        file = request.files['image']
        is_valid, error_msg, image = validate_image(file)
        
        if not is_valid:
            return jsonify({
                'error': 'Invalid image',
                'message': error_msg
            }), 400
        
        # ==================== PREPROCESSING ====================
        # Preprocess image
        img_array = preprocess_image(image)
        
        # Use average values from CSV for tabular input
        avg_values = df[tab_features_model].mean()
        tab_input = np.array([[
            avg_values['Humidity'],
            avg_values['Temperature'],
            avg_values['Soil_pH']
        ]])
        
        # Scale tabular input
        tab_input_scaled = scaler.transform(tab_input)
        
        # ==================== PREDICTION ====================
        pred_probs = model.predict([img_array, tab_input_scaled], verbose=0)
        
        # EXCLUDE "Tomato Spotted Wilt" from predictions
        excluded_diseases = ["Tomato Spotted Wilt"]
        
        # Get all disease names
        all_diseases = disease_le.inverse_transform(np.arange(len(pred_probs[0])))
        
        # Find indices of excluded diseases
        excluded_indices = []
        for i, disease in enumerate(all_diseases):
            if disease in excluded_diseases:
                excluded_indices.append(i)
        
        # Create filtered probabilities
        pred_probs_filtered = pred_probs[0].copy()
        for idx in excluded_indices:
            pred_probs_filtered[idx] = -1
        
        # Get prediction excluding the diseases
        pred_class = np.argmax(pred_probs_filtered)
        pred_label = disease_le.inverse_transform([pred_class])[0]
        confidence = float(pred_probs[0][pred_class] * 100)
        
        # Get top 5 predictions (excluding excluded diseases)
        sorted_indices = np.argsort(pred_probs_filtered)[::-1]
        top_predictions = []
        for idx in sorted_indices:
            disease_name = disease_le.inverse_transform([idx])[0]
            if disease_name not in excluded_diseases and len(top_predictions) < 5:
                top_predictions.append({
                    'disease': disease_name,
                    'confidence': float(pred_probs[0][idx] * 100)
                })
        
        # Get top 3 for backward compatibility
        top_3_predictions = top_predictions[:3]
        
        # ==================== RESPONSE ====================
        # Add health assessment based on confidence
        health_status = None
        if confidence < 50:
            health_status = {
                'status': 'possibly_healthy',
                'message': 'Low confidence in disease prediction. Plant may be healthy or have symptoms not in training data.',
                'recommendation': 'Manual inspection recommended'
            }
        elif confidence < 70:
            health_status = {
                'status': 'uncertain',
                'message': 'Medium confidence. Disease symptoms may be unclear or early stage.',
                'recommendation': 'Monitor plant and re-check in a few days'
            }
        else:
            health_status = {
                'status': 'disease_detected',
                'message': 'High confidence in disease detection.',
                'recommendation': 'Take appropriate treatment measures'
            }
        
        response = {
            'success': True,
            'prediction': {
                'disease': pred_label,
                'confidence': round(confidence, 2),
                'confidence_level': 'high' if confidence >= 80 else 'medium' if confidence >= 60 else 'low'
            },
            'health_assessment': health_status,
            'top_predictions': top_3_predictions,
            'all_predictions': top_predictions,  # Show top 5
            'input_data': {
                'image_size': image.size,
                'tabular_data_source': 'average_values',
                'humidity': round(avg_values['Humidity'], 2),
                'temperature': round(avg_values['Temperature'], 2),
                'soil_ph': round(avg_values['Soil_pH'], 2)
            },
            'model_info': {
                'note': 'This model was trained only on diseased plants. It cannot detect healthy plants.',
                'trained_classes': len(disease_le.classes_),
                'has_healthy_class': False
            },
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response), 200
    
    except Exception as e:
        print(f"❌ Prediction error: {str(e)}")
        traceback.print_exc()
        return jsonify({
            'error': 'Prediction failed',
            'message': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500


@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    return jsonify({
        'error': 'File too large',
        'message': f'Maximum file size is {app.config["MAX_CONTENT_LENGTH"] / (1024 * 1024)}MB'
    }), 413


@app.errorhandler(500)
def internal_error(e):
    """Handle internal server error"""
    return jsonify({
        'error': 'Internal server error',
        'message': str(e)
    }), 500


# ==================== STARTUP ====================
@app.before_request
def initialize():
    """Initialize model and data before first request"""
    global model
    if model is None:
        success = load_model_and_data()
        if not success:
            return jsonify({
                'error': 'Failed to initialize model',
                'message': 'Server initialization failed'
            }), 500


if __name__ == '__main__':
    print("="*60)
    print("🚀 Starting Tomato Disease Detection API")
    print("="*60)
    
    # Load model and data
    if load_model_and_data():
        print("\n✅ Server ready!")
        print("📡 API Endpoints:")
        print("   - GET  /health       : Health check")
        print("   - GET  /info         : Model information")
        print("   - POST /predict      : Prediction with custom tabular data")
        print("   - POST /predict-auto : Prediction with automatic tabular data")
        print("\n🌐 Starting Flask server...")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("\n❌ Failed to initialize server")