from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io, os, numpy as np, pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelEncoder, StandardScaler
import traceback
from datetime import datetime

app = Flask(__name__)

# -------------------- CORS --------------------
CORS(app, resources={r"/*": {"origins": "*"}})  # allow all origins

# -------------------- CONFIG --------------------
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# -------------------- PATHS --------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(script_dir, "best_fusion_crop_final.h5")
CSV_PATH = os.path.join(script_dir, "DATASET", "tomato_disease_dataset .csv")

# -------------------- GLOBALS --------------------
model = None
df = None
scaler = None
disease_le = None
tab_features_model = ['Humidity', 'Temperature', 'Soil_pH']

# -------------------- HELPERS --------------------
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def validate_image(file):
    if not file or file.filename == '' or not allowed_file(file.filename):
        return False, "Invalid or missing file", None
    try:
        img = Image.open(file.stream)
        img.verify()  # validate image
        file.seek(0)
        img = Image.open(file.stream).convert('RGB')
        return True, None, img
    except Exception as e:
        return False, f"Invalid image file: {str(e)}", None

def preprocess_image(image, target_size=(224, 224)):
    image = image.resize(target_size)
    arr = img_to_array(image) / 255.0
    return np.expand_dims(arr, axis=0)

def load_model_and_data():
    global model, df, scaler, disease_le
    try:
        print("🔹 Loading model...")
        model = load_model(MODEL_PATH)
        print("✅ Model loaded")

        df = pd.read_csv(CSV_PATH)
        df.rename(columns={'Humidity (%)': 'Humidity', 'Temperature (°C)': 'Temperature', 'Soil pH': 'Soil_pH'}, inplace=True)
        df = df.dropna(subset=tab_features_model + ['Disease Name'])
        scaler = StandardScaler()
        scaler.fit(df[tab_features_model])
        disease_le = LabelEncoder()
        disease_le.fit(df['Disease Name'])
        print(f"✅ Data loaded with {len(disease_le.classes_)} disease classes")
        return True
    except Exception as e:
        print("❌ Failed to load model/data:", e)
        traceback.print_exc()
        return False

# -------------------- ENDPOINTS --------------------
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'running',
        'model_loaded': model is not None,
        'data_loaded': df is not None,
        'timestamp': datetime.now().isoformat()
    }), 200

@app.route('/predict-auto', methods=['POST'])
def predict_auto():
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500

        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400

        file = request.files['image']
        valid, msg, img = validate_image(file)
        if not valid:
            return jsonify({'error': msg}), 400

        img_array = preprocess_image(img)
        avg_values = df[tab_features_model].mean()
        tab_input = np.array([[avg_values['Humidity'], avg_values['Temperature'], avg_values['Soil_pH']]])
        tab_input_scaled = scaler.transform(tab_input)

        pred_probs = model.predict([img_array, tab_input_scaled], verbose=0)
        pred_class = np.argmax(pred_probs[0])
        pred_label = disease_le.inverse_transform([pred_class])[0]
        confidence = float(pred_probs[0][pred_class] * 100)

        return jsonify({
            'prediction': {'disease': pred_label, 'confidence': round(confidence, 2)},
            'timestamp': datetime.now().isoformat()
        }), 200

    except Exception as e:
        print("❌ Prediction error:", e)
        traceback.print_exc()
        return jsonify({'error': str(e), 'timestamp': datetime.now().isoformat()}), 500

# -------------------- STARTUP --------------------
@app.before_first_request
def init_app():
    if load_model_and_data():
        print("✅ Model and data ready")
    else:
        print("❌ Failed to initialize model/data")

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
