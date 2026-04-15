import gradio as gr
from PIL import Image
import numpy as np
import tensorflow as tf

# ============================================================
# 38 classes - PlantVillage Dataset
# ============================================================
CLASS_NAMES = [
    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy",
    "Blueberry___healthy", "Cherry_(including_sour)___Powdery_mildew", "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight", "Corn_(maize)___healthy",
    "Grape___Black_rot", "Grape___Esca_(Black_Measles)", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)", "Peach___Bacterial_spot", "Peach___healthy",
    "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy",
    "Potato___Early_blight", "Potato___Late_blight", "Potato___healthy",
    "Raspberry___healthy", "Soybean___healthy", "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch", "Strawberry___healthy",
    "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight", "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot", "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot", "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus", "Tomato___healthy"
]

# ============================================================
# Model Loading (Lazy Loading for HF Spaces Efficiency)
# ============================================================
DISEASE_MODEL = None
GATEKEEPER_MODEL = None

def load_models():
    global DISEASE_MODEL, GATEKEEPER_MODEL
    # 1. Load the Disease Model
    if DISEASE_MODEL is None:
        try:
            DISEASE_MODEL = tf.keras.models.load_model("plant_disease_model_optimized.keras")
            print("✅ Specialist Model loaded (.keras)")
        except:
            DISEASE_MODEL = tf.keras.models.load_model("plant_disease_model_optimized.h5")
            print("✅ Specialist Model loaded (.h5)")
    
    # 2. Load the Gatekeeper (MobileNetV2)
    if GATEKEEPER_MODEL is None:
        GATEKEEPER_MODEL = tf.keras.applications.MobileNetV2(weights="imagenet")
        print("✅ Gatekeeper Model loaded (MobileNetV2)")

def is_actually_a_leaf(img):
    """
    Uses ImageNet-trained MobileNetV2 to check if the image is a leaf/plant.
    Returns: (is_leaf, reason, top_general_labels)
    """
    img_gate = img.resize((224, 224))
    x = tf.keras.applications.mobilenet_v2.preprocess_input(np.array(img_gate))
    x = np.expand_dims(x, axis=0)
    
    preds = GATEKEEPER_MODEL.predict(x, verbose=0)
    decoded = tf.keras.applications.mobilenet_v2.decode_predictions(preds, top=10)[0]
    
    top_labels = [label.lower() for (_, label, _) in decoded]
    top_conf = decoded[0][2]

    PLANT_KEYWORDS = ['leaf', 'foliage', 'plant', 'corn', 'buckeye', 'vegetable', 'fruit']
    REJECT_KEYWORDS = ['plate', 'dish', 'food', 'pizza', 'burger', 'dog', 'cat', 'car', 'person']

    if any(r in top_labels[0] for r in REJECT_KEYWORDS):
        return False, f"Detected: {top_labels[0].replace('_', ' ')}. Not a plant leaf.", top_labels

    is_plant = any(any(k in label for k in PLANT_KEYWORDS) for label in top_labels)
    
    if not is_plant and top_conf > 0.30:
        return False, f"Detected: {top_labels[0].replace('_', ' ')}. Not a plant leaf.", top_labels
        
    return True, "", top_labels

def get_prediction(img):
    if img is None:
        return "No image", "Please upload an image."
        
    load_models()
    
    is_leaf, reason, general_labels = is_actually_a_leaf(img)
    if not is_leaf:
        return "❌ Non-leaf detected", f"❌ {reason}"

    img_resized = img.resize((224, 224))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = DISEASE_MODEL.predict(img_array, verbose=0)[0]
    sorted_idx = np.argsort(predictions)[::-1]
    
    top_class = CLASS_NAMES[sorted_idx[0]]
    top_conf = float(predictions[sorted_idx[0]])
    
    # Format Top 5 for Textbox
    top5_idx = sorted_idx[:5]
    results_text = "📊 Top 5 Predictions:\n" + "\n".join([f"• {CLASS_NAMES[i]}: {predictions[i]*100:.1f}%" for i in top5_idx])

    plant, condition = top_class.split("___") if "___" in top_class else (top_class, "Unknown")
    plant = plant.replace("_", " ")
    condition = condition.replace("_", " ")

    # Uncertainty logic (Cross-verification)
    predicted_plant_base = plant.lower().split("(")[0].strip()
    top2_plants = [CLASS_NAMES[i].split("___")[0].lower() for i in sorted_idx[:2]]
    species_agreement = top2_plants[0] == top2_plants[1]
    
    # Dynamic messaging
    if top_conf < 0.70:
        status = f"⚠️ Low confidence ({top_conf*100:.1f}%) — your plant or disease may not be in our 38 supported classes. Please check the supported list.\n\n"
        status += f"❓ Possible Match: {plant} — {condition}"
    elif not species_agreement and top_conf < 0.90:
        status = f"❓ Possible Match: {plant} — {condition}\n\n"
        status += f"⚠️ Warning: The system is uncertain. This plant (like Avocado) may not be supported."
    else:
        status = f"✅ {plant} — Healthy" if "healthy" in condition.lower() else f"⚠️ {plant} — {condition} detected"
        status += f" ({top_conf*100:.1f}% confidence)"

    return results_text, status

# ============================================================
# Gradio Interface
# ============================================================
custom_css = """
.green-btn { background-color: #28a745 !important; color: white !important; border: none !important; }
.green-btn:hover { background-color: #218838 !important; }
.warning-banner {
    background-color: #fff3cd !important;
    color: #856404 !important;
    padding: 15px !important;
    border-radius: 8px !important;
    border: 1px solid #ffeeba !important;
    margin-bottom: 20px !important;
    font-weight: bold !important;
    text-align: center !important;
}
/* Style for component labels */
label span {
    background-color: #d4edda !important;
    color: #155724 !important;
    padding: 4px 12px !important;
    border-radius: 6px !important;
    font-weight: bold !important;
    display: inline-block !important;
    margin-bottom: 4px !important;
}
"""

grouped_plants = {}
for name in CLASS_NAMES:
    p = name.split("___")[0].replace("_", " ")
    if p not in grouped_plants: grouped_plants[p] = []
    cond = name.split("___")[1].replace("_", " ") if "___" in name else "Unknown"
    grouped_plants[p].append(cond)

supported_list_md = ""
for plant, diseases in grouped_plants.items():
    supported_list_md += f"**{plant}:** {', '.join(diseases)}\n\n"

with gr.Blocks(theme=gr.themes.Soft(), css=custom_css, title="🌿 Smart Plant Doctor") as demo:
    gr.HTML("""
    <div class="warning-banner">
        ⚠️ This model supports 38 classes only from the PlantVillage dataset. 
        Results may be inaccurate for plants or diseases not in this list.
    </div>
    """)
    
    gr.Markdown("""
    # 🌿 Plant Disease Detector
    
    Instantly detect plant diseases using deep learning. This tool analyzes leaf images to identify health issues across 38 specific plant-disease categories.
    
    **How to use:**
    1. **Upload** a clear, well-lit image of a single plant leaf.
    2. **Wait** for the AI to verify if it's a leaf.
    3. **Review** the diagnosis and confidence level.
    """)
    
    with gr.Accordion("📋 See all 38 supported plants", open=False):
        gr.Markdown(supported_list_md)
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="📷 Upload Leaf Image")
            predict_btn = gr.Button("🔍 Analyze Leaf", variant="primary", elem_classes="green-btn")
        with gr.Column():
            status_output = gr.Textbox(label="🎯 Diagnosis", interactive=False, lines=5)
            label_output = gr.Textbox(label="📊 Probabilities", interactive=False, lines=6)

    predict_btn.click(fn=get_prediction, inputs=input_image, outputs=[label_output, status_output])

if __name__ == "__main__":
    demo.launch()
