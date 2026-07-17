"""
SynapseVision - Flask app for classifying brain MRI scans.

Loads a VGG16-based model and serves a simple upload UI. If no model is
available it falls back to a simulated prediction so the UI can still be
demoed, but this is always clearly flagged - it's never shown as a real
result.
"""

from flask import Flask, render_template, request, send_from_directory
from werkzeug.utils import secure_filename
import numpy as np
import os
import base64
import requests
from dotenv import load_dotenv
from PIL import Image

load_dotenv()

app = Flask(__name__)

# ---------------------------------------------------------------------
# Load the trained model
# ---------------------------------------------------------------------
HAS_TENSORFLOW = False
model = None

try:
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.image import load_img, img_to_array
    HAS_TENSORFLOW = True
except ImportError:
    print("Warning: TensorFlow or Keras is not installed. Using simulated AI predictions.")

if HAS_TENSORFLOW:
    try:
        weights_path = 'models/model.weights.h5'
        legacy_full_model_path = 'models/model.h5'

        if os.path.exists(weights_path):
            # We only ship the weights, not a full saved model - loading a
            # full .h5 model tends to break across Keras versions (mismatched
            # layer configs etc), so we rebuild the architecture here and
            # load the numbers into it instead. More boilerplate, but it
            # actually works across versions.
            from tensorflow.keras.applications import VGG16
            from tensorflow.keras import layers, models as keras_models

            IMAGE_SIZE = 64  # this has to match what the model was trained on
            base_model = VGG16(weights=None, include_top=False,
                                input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
            model = keras_models.Sequential([
                base_model,
                layers.Flatten(),
                layers.Dense(256, activation='relu'),
                layers.Dropout(0.5),
                layers.Dense(4, activation='softmax'),
            ])
            model.load_weights(weights_path)
            print("Loaded VGG16 weights from models/model.weights.h5")
        elif os.path.exists(legacy_full_model_path):
            model = load_model(legacy_full_model_path)
            print("Loaded VGG16 model from models/model.h5")
        else:
            print(f"No model file found (checked '{weights_path}' and "
                  f"'{legacy_full_model_path}'). Falling back to simulated predictions.")
            HAS_TENSORFLOW = False
    except Exception as e:
        print(f"Error loading model: {e}. Falling back to simulated predictions.")
        HAS_TENSORFLOW = False

# Class order matches how Keras' flow_from_directory sorts the training
# folders (alphabetical). If you retrain this, double check against your
# own train_generator.class_indices before trusting this list.
CLASS_LABELS = ['glioma', 'meningioma', 'notumor', 'pituitary']

UPLOAD_FOLDER = './uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tif', 'tiff'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def simulate_prediction(image_path):
    # Used when there's no model loaded - picks a class based on the
    # filename so the demo looks half-reasonable, but this is never
    # presented as a real prediction (see is_demo below).
    filename = os.path.basename(image_path).lower()

    if 'glioma' in filename or 'gl' in filename:
        chosen_class, base_conf = 'glioma', 0.942
    elif 'pituitary' in filename or 'pi' in filename:
        chosen_class, base_conf = 'pituitary', 0.968
    elif 'meningioma' in filename or 'me' in filename:
        chosen_class, base_conf = 'meningioma', 0.915
    elif any(k in filename for k in ('no', 'healthy', 'normal', 'notumor')):
        chosen_class, base_conf = 'notumor', 0.984
    else:
        hash_val = sum(ord(c) for c in filename)
        chosen_class = CLASS_LABELS[hash_val % len(CLASS_LABELS)]
        base_conf = 0.85 + ((hash_val % 14) / 100.0)

    return chosen_class, base_conf


def predict_tumor(image_path):
    """Returns (result_text, confidence, is_demo_mode)."""
    if HAS_TENSORFLOW and model is not None:
        try:
            IMAGE_SIZE = 64
            img = load_img(image_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
            img_array = img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            predictions = model.predict(img_array)
            predicted_class_index = np.argmax(predictions, axis=1)[0]
            confidence_score = np.max(predictions, axis=1)[0]
            predicted_class = CLASS_LABELS[predicted_class_index]
            is_demo = False
        except Exception as e:
            print(f"Prediction failed: {e}. Falling back to simulation.")
            predicted_class, confidence_score = simulate_prediction(image_path)
            is_demo = True
    else:
        predicted_class, confidence_score = simulate_prediction(image_path)
        is_demo = True

    result_text = "No Tumor" if predicted_class == 'notumor' else f"Tumor: {predicted_class}"
    return result_text, confidence_score, is_demo


# Fireworks AI gives us a vision-capable, OpenAI-compatible chat endpoint
# for turning the raw classification into a short explanation. Get a key
# at https://fireworks.ai and put it in .env as FIREWORKS_API_KEY.
FIREWORKS_API_KEY = os.getenv('FIREWORKS_API_KEY')
FIREWORKS_MODEL = os.getenv('FIREWORKS_MODEL', 'accounts/fireworks/models/kimi-k2p5')
FIREWORKS_URL = "https://api.fireworks.ai/inference/v1/chat/completions"


def encode_image_base64(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def generate_clinical_insight(image_path, result_text, confidence_score, is_demo_mode):
    if is_demo_mode:
        # Don't bother calling the LLM on top of a fake classification -
        # just say plainly that this isn't real.
        tumor_type = result_text.replace("Tumor: ", "").strip()
        if result_text == "No Tumor":
            return (
                "SIMULATED RESULT - no trained model was loaded, so this "
                "classification and insight are placeholders for UI demo "
                "purposes only, not a real analysis."
            )
        return (
            f"SIMULATED RESULT - no trained model was loaded, so the "
            f"'{tumor_type}' classification above is a placeholder for UI "
            "demo purposes only, not a real analysis."
        )

    if not FIREWORKS_API_KEY:
        return (
            "AI insight unavailable: no FIREWORKS_API_KEY set in your .env "
            "file. The tumor classification above still comes from your "
            "real trained model - only the natural-language explanation "
            "step is skipped. Get a free key at https://fireworks.ai."
        )

    try:
        image_b64 = encode_image_base64(image_path)
        mime = "image/png" if image_path.lower().endswith("png") else "image/jpeg"

        prompt = (
            f"You are an AI assistant helping explain outputs of a "
            f"medical-imaging classifier. A local Convolutional Neural "
            f"Network has classified this brain scan as '{result_text}' "
            f"with a confidence score of {confidence_score}%. In 2-3 "
            f"concise, professional sentences, explain what this "
            f"classification means, note typical next steps, and end with "
            f"a clear disclaimer that this must be verified by a licensed "
            f"radiologist or specialist."
        )

        response = requests.post(
            FIREWORKS_URL,
            headers={
                "Authorization": f"Bearer {FIREWORKS_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": FIREWORKS_MODEL,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:{mime};base64,{image_b64}"},
                            },
                        ],
                    }
                ],
                "max_tokens": 300,
            },
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Error generating AI insight from Fireworks AI: {str(e)}"


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files.get('file')

        if not file or file.filename == '':
            return render_template('index.html', result=None, error="No file selected.")

        if not allowed_file(file.filename):
            return render_template(
                'index.html', result=None,
                error="Unsupported file type. Please upload a PNG or JPG image."
            )

        filename = secure_filename(file.filename)  # avoid path traversal / overwrite tricks
        file_location = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_location)

        result, confidence, is_demo = predict_tumor(file_location)
        rounded_confidence = round(float(confidence) * 100, 2)

        ai_insight = generate_clinical_insight(file_location, result, rounded_confidence, is_demo)

        return render_template(
            'index.html',
            result=result,
            confidence=rounded_confidence,
            file_path=f'/uploads/{filename}',
            ai_insight=ai_insight,
            is_demo_mode=is_demo,
        )

    return render_template('index.html', result=None)


@app.route('/uploads/<filename>')
def get_uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], secure_filename(filename))


if __name__ == '__main__':
    debug_mode = os.getenv('FLASK_DEBUG', 'false').lower() == 'true'
    app.run(debug=debug_mode)
