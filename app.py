from flask import Flask, render_template, request, redirect, url_for, flash
import os
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
import numpy as np
import logging
import cv2

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

# Configuration
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5MB limit

# Load model and detect input shape
try:
    model = load_model('model.h5', compile=False)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    logger.info("Model loaded successfully")

    if hasattr(model.layers[0], 'input_shape') and model.layers[0].input_shape:
        input_shape = model.layers[0].input_shape[1:]  # Exclude batch dimension
        logger.info(f"Model expects input shape: {input_shape}")
    else:
        input_shape = (160,160, 5)  # Fallback to default
        logger.warning(f"Could not detect model input shape, using default: {input_shape}")

except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")
    exit(1)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path, target_size):
    """Preprocess image to match model's expected 5-channel input"""
    try:
        # Read image using OpenCV (BGR format)
        bgr_img = cv2.imread(image_path)
        if bgr_img is None:
            raise ValueError("Could not read image")

        # Resize to (width, height)
        bgr_img = cv2.resize(bgr_img, (target_size[1], target_size[0]))

        # Convert BGR to RGB
        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        rgb_img = rgb_img.astype('float32') / 255.0

        # Generate grayscale
        gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY).astype('float32') / 255.0

        # Generate edges using Canny
        edges = cv2.Canny((gray * 255).astype('uint8'), 100, 200).astype('float32') / 255.0

        # Expand gray and edge to (H, W, 1)
        gray = np.expand_dims(gray, axis=-1)
        edges = np.expand_dims(edges, axis=-1)

        # Stack to form 5-channel input
        img_5ch = np.concatenate([rgb_img, gray, edges], axis=-1)

        logger.info(f"Preprocessed image shape: {img_5ch.shape}")

        return np.expand_dims(img_5ch, axis=0)  # Add batch dimension

    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        return None

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file selected')
            return redirect(request.url)

        file = request.files['file']

        if file.filename == '':
            flash('No file selected')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            try:
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)

                # Preprocess the uploaded image
                processed_img = preprocess_image(filepath, input_shape)
                if processed_img is None:
                    flash('Error processing image')
                    return redirect(request.url)

                # Make prediction
                prediction = model.predict(processed_img)
                confidence = float(prediction[0][0])

                # Classify result
                if confidence > 0.5:
                    result = "FAKE (Deepfake)"
                    confidence_percent = round(confidence * 100, 2)
                else:
                    result = "REAL"
                    confidence_percent = round((1 - confidence) * 100, 2)

                return render_template('index.html',
                                       result=result,
                                       confidence=confidence_percent,
                                       filename=filename)

            except Exception as e:
                logger.error(f"Error during prediction: {str(e)}")
                flash('An error occurred during processing')
                return redirect(request.url)

    return render_template('index.html')

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(host='0.0.0.0', port=5000, debug=True)
