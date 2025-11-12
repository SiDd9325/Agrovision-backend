import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score,
    precision_score, recall_score, f1_score, roc_curve, auc
)
import matplotlib.pyplot as plt
import seaborn as sns
import os

# -------------------- CONFIGURATION --------------------
# Set to True to evaluate on test dataset, False for single image prediction
EVALUATE_ON_TEST_SET = True  # Change to True for full evaluation

# -------------------- PATHS --------------------
# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(script_dir, "best_fusion_crop_final.h5")
csv_path   = os.path.join(script_dir, "DATASET", "tomato_disease_dataset .csv")  # Note: filename has a space before .csv
img_path   = os.path.join(script_dir, "DATASET", "tomato", "train", "Tomato___Bacterial_spot", "34385d7a-a724-4577-898b-dc9b9deb8ed9___GCREC_Bact.Sp 6098.JPG")
test_dir   = os.path.join(script_dir, "DATASET", "tomato", "test")

# -------------------- LOAD MODEL --------------------
print("🔹 Loading trained fusion model...")
model = load_model(model_path)

# -------------------- HELPER FUNCTIONS --------------------
def normalize_label(name):
    """Normalize disease label to match CSV format"""
    return name.replace("Tomato___", "").replace("_", " ").strip().lower()

def load_test_images(folder, max_samples_per_class=None):
    """Load test images with their labels"""
    X_img, y_labels = [], []
    for disease_folder in os.listdir(folder):
        folder_path = os.path.join(folder, disease_folder)
        if not os.path.isdir(folder_path):
            continue
        label = normalize_label(disease_folder)
        count = 0
        for f in os.listdir(folder_path):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                if max_samples_per_class and count >= max_samples_per_class:
                    break
                img_path = os.path.join(folder_path, f)
                try:
                    img = load_img(img_path, target_size=(128, 128))
                    img = img_to_array(img) / 255.0
                    X_img.append(img)
                    y_labels.append(label)
                    count += 1
                except Exception as e:
                    continue
    return np.array(X_img), np.array(y_labels)

def get_tabular_data_for_labels(csv_df, labels, features, scaler):
    """Get tabular data for given labels"""
    rows = []
    for lbl in labels:
        match = csv_df[csv_df['Disease_Name_Norm'] == lbl]
        if not match.empty:
            rows.append(match[features].iloc[0].fillna(0).values)
        else:
            # Use mean values if no match found
            rows.append(csv_df[features].mean().fillna(0).values)
    tab_array = np.array(rows)
    return scaler.transform(tab_array)

# -------------------- LOAD & PREPARE CSV DATA --------------------
df = pd.read_csv(csv_path)

# Standardize column names
df.rename(columns={
    'Disease Name': 'Disease_Name',
    'Leaf Color': 'Leaf_Color',
    'Spots Present': 'Spots_Present',
    'Humidity (%)': 'Humidity',
    'Temperature (°C)': 'Temperature',
    'Soil pH': 'Soil_pH'
}, inplace=True)

# Create normalized disease name for matching
df['Disease_Name_Norm'] = df['Disease_Name'].str.strip().str.lower()

# Drop rows with missing values in relevant columns
tab_features_model = ['Humidity', 'Temperature', 'Soil_pH']  # Only numeric features used by model
df = df.dropna(subset=tab_features_model + ['Disease_Name'])

# Normalize numeric tabular features
scaler = StandardScaler()
df[tab_features_model] = scaler.fit_transform(df[tab_features_model])

# Encode target labels
disease_le = LabelEncoder()
df['Disease_Name_Encoded'] = disease_le.fit_transform(df['Disease_Name'])

# -------------------- EVALUATION MODE --------------------
if EVALUATE_ON_TEST_SET:
    print("\n📊 Loading test dataset for evaluation...")
    # Load test images (limit to 50 per class for faster evaluation, remove limit for full evaluation)
    X_test_img, y_test_labels = load_test_images(test_dir, max_samples_per_class=50)
    print(f"Loaded {len(X_test_img)} test images")
    
    # Get tabular data for test images
    X_test_tab = get_tabular_data_for_labels(df, y_test_labels, tab_features_model, scaler)
    
    # Encode test labels
    y_test_encoded = []
    valid_indices = []
    for idx, lbl in enumerate(y_test_labels):
        match = df[df['Disease_Name_Norm'] == lbl]
        if not match.empty:
            y_test_encoded.append(match['Disease_Name_Encoded'].iloc[0])
            valid_indices.append(idx)
        else:
            print(f"Warning: Label '{lbl}' not found in CSV, skipping sample {idx}")
    
    # Filter out samples with missing labels
    if len(valid_indices) < len(y_test_labels):
        X_test_img = X_test_img[valid_indices]
        X_test_tab = X_test_tab[valid_indices]
        print(f"Using {len(valid_indices)} valid samples out of {len(y_test_labels)} total")
    
    y_test_encoded = np.array(y_test_encoded)
    
    # Get unique classes present in test data
    unique_test_classes = np.unique(y_test_encoded)
    print(f"Test set contains {len(unique_test_classes)} unique classes: {unique_test_classes}")
    
    print("🔹 Making predictions on test set...")
    # Make predictions in batches
    batch_size = 32
    y_pred_probs = []
    for i in range(0, len(X_test_img), batch_size):
        batch_img = X_test_img[i:i+batch_size]
        batch_tab = X_test_tab[i:i+batch_size]
        batch_probs = model.predict([batch_img, batch_tab], verbose=0)
        y_pred_probs.append(batch_probs)
    y_pred_probs = np.vstack(y_pred_probs)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Get all possible classes from the model
    all_classes = np.arange(len(disease_le.classes_))
    present_classes = np.unique(np.concatenate([y_test_encoded, y_pred]))
    
    # -------------------- CALCULATE METRICS --------------------
    print("\n" + "="*60)
    print("📈 COMPREHENSIVE EVALUATION METRICS")
    print("="*60)
    
    # Accuracy
    accuracy = accuracy_score(y_test_encoded, y_pred)
    print(f"\n✅ Accuracy: {accuracy*100:.2f}%")
    
    # Precision, Recall, F1-Score (macro and weighted averages)
    precision_macro = precision_score(y_test_encoded, y_pred, average='macro', zero_division=0, labels=present_classes)
    precision_weighted = precision_score(y_test_encoded, y_pred, average='weighted', zero_division=0, labels=present_classes)
    recall_macro = recall_score(y_test_encoded, y_pred, average='macro', zero_division=0, labels=present_classes)
    recall_weighted = recall_score(y_test_encoded, y_pred, average='weighted', zero_division=0, labels=present_classes)
    f1_macro = f1_score(y_test_encoded, y_pred, average='macro', zero_division=0, labels=present_classes)
    f1_weighted = f1_score(y_test_encoded, y_pred, average='weighted', zero_division=0, labels=present_classes)
    
    print(f"\n📊 Precision (Macro): {precision_macro:.4f}")
    print(f"📊 Precision (Weighted): {precision_weighted:.4f}")
    print(f"📊 Recall (Macro): {recall_macro:.4f}")
    print(f"📊 Recall (Weighted): {recall_weighted:.4f}")
    print(f"📊 F1-Score (Macro): {f1_macro:.4f}")
    print(f"📊 F1-Score (Weighted): {f1_weighted:.4f}")
    
    # Classification Report - only include classes present in test data
    disease_names = disease_le.classes_
    print("\n" + "="*60)
    print("📋 CLASSIFICATION REPORT")
    print("="*60)
    # Use labels parameter to specify which classes to include
    print(classification_report(y_test_encoded, y_pred, 
                                labels=present_classes,
                                target_names=[disease_names[i] for i in present_classes], 
                                zero_division=0))
    
    # Confusion Matrix - include all classes that appear in test or predictions
    cm = confusion_matrix(y_test_encoded, y_pred, labels=present_classes)
    
    # Plot Confusion Matrix
    plt.figure(figsize=(12, 10))
    present_disease_names = [disease_names[i] for i in present_classes]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=present_disease_names, yticklabels=present_disease_names,
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()
    
    # ROC Curve (for multiclass) - use all model classes for binarization
    n_classes = len(disease_names)
    y_test_binarized = label_binarize(y_test_encoded, classes=np.arange(n_classes))
    
    # Calculate ROC for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_pred_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Calculate micro-average ROC
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test_binarized.ravel(), y_pred_probs.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    # Calculate macro-average ROC
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    
    print(f"\n📈 ROC AUC Scores:")
    print(f"   Micro-average: {roc_auc['micro']:.4f}")
    print(f"   Macro-average: {roc_auc['macro']:.4f}")
    # Only print AUC for classes present in test data
    for i in present_classes:
        print(f"   {disease_names[i]}: {roc_auc[i]:.4f}")
    
    # Plot ROC Curves
    plt.figure(figsize=(12, 8))
    # Create a finite list of colors (not a cycle)
    color_list = ['aqua', 'darkorange', 'cornflowerblue', 'red', 'green', 
                  'purple', 'brown', 'pink', 'gray', 'olive', 'cyan', 
                  'magenta', 'yellow', 'lime', 'teal']
    
    # Plot micro and macro averages
    plt.plot(fpr["micro"], tpr["micro"],
             label=f'Micro-average ROC (AUC = {roc_auc["micro"]:.2f})',
             color='deeppink', linestyle='--', linewidth=2)
    plt.plot(fpr["macro"], tpr["macro"],
             label=f'Macro-average ROC (AUC = {roc_auc["macro"]:.2f})',
             color='navy', linestyle='--', linewidth=2)
    
    # Plot ROC for each class present in test data
    for idx, i in enumerate(present_classes):
        plt.plot(fpr[i], tpr[i], color=color_list[idx % len(color_list)], lw=1.5,
                 label=f'{disease_names[i]} (AUC = {roc_auc[i]:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=1, label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Multi-Class ROC Curves', fontsize=16, fontweight='bold')
    plt.legend(loc="lower right", fontsize=9)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print("\n✅ Evaluation complete!")

# -------------------- SINGLE IMAGE PREDICTION MODE --------------------
else:
    print("\n🖼️  Single Image Prediction Mode")
    print("="*60)
    
    # Load single image
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = np.expand_dims(image.img_to_array(img) / 255.0, axis=0)
    
    # Pick a random sample from CSV for tabular input (or you can use your own values)
    sample = df.sample(1, random_state=42)
    tab_input = sample[tab_features_model].values  # shape (1,3)
    
    # -------------------- PREDICTION --------------------
    pred_probs = model.predict([img_array, tab_input], verbose=0)
    pred_class = np.argmax(pred_probs, axis=1)[0]
    pred_label = disease_le.inverse_transform([pred_class])[0]
    
    print("\n🩺 Prediction Results")
    print("-----------------------")
    print(f"Predicted Disease : {pred_label}")
    print(f"Confidence        : {np.max(pred_probs)*100:.2f}%")
    print(f"\nAssociated Tabular Input:")
    print(sample[tab_features_model].to_string(index=False))
    
    # -------------------- PLOT CLASS PROBABILITIES --------------------
    all_probs = pred_probs[0]
    disease_names = disease_le.inverse_transform(np.arange(len(all_probs)))
    
    plt.figure(figsize=(10, 6))
    plt.barh(disease_names, all_probs, color='skyblue')
    plt.xlabel("Probability", fontsize=12)
    plt.ylabel("Disease Class", fontsize=12)
    plt.title("Predicted Disease Class Probabilities", fontsize=14, fontweight='bold')
    plt.xlim(0, 1)
    for i, v in enumerate(all_probs):
        plt.text(v + 0.01, i, f"{v*100:.2f}%", va='center', fontsize=9)
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print("\n💡 Tip: Set EVALUATE_ON_TEST_SET = True to run full evaluation metrics")
