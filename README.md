Deepfake Detection with Frequency Domain CNN
This project implements a Convolutional Neural Network (CNN) to detect deepfake images (real vs. fake faces) using frequency domain preprocessing. By leveraging the "140k Real and Fake Faces" dataset from Kaggle, the model incorporates Discrete Fourier Transform (DFT) to enhance detection accuracy.
Features

Dataset: Subset of 29,400 training, 6,300 validation, and 6,300 test images (balanced real/fake).
Preprocessing: Adds frequency channels (magnitude, phase) to RGB images for 5-channel inputs.
Model: Custom CNN with Conv2D, BatchNorm, LeakyReLU, Dropout, and L2 regularization.
Training: Uses class weights, early stopping, learning rate reduction, and checkpointing.
Evaluation: Includes classification reports, confusion matrices, and ROC curves.
Output: Saves model as my_model.h5 (HDF5 format).

Prerequisites

Google Colab with GPU (T4 recommended).
Kaggle account and API token (kaggle.json).
Python 3.x environment.

Installation
In Google Colab, the notebook installs dependencies automatically. For local setup:
pip install tensorflow numpy matplotlib opencv-python scikit-learn seaborn kaggle

Place kaggle.json in ~/.kaggle/ and set permissions: chmod 600 ~/.kaggle/kaggle.json.
Usage

Open deepfake_detection.ipynb in Google Colab.
Upload kaggle.json when prompted.
Run cells sequentially to:
Download and prepare the dataset.
Train the model (up to 50 epochs with early stopping).
View evaluation metrics and plots.
Download my_model.h5.


For inference:model = tf.keras.models.load_model('my_model.h5', custom_objects={'LeakyReLU': LeakyReLU})
preprocessed_image = add_frequency_channels(image)
prediction = model.predict(preprocessed_image)  # Threshold: 0.5



Directory Structure
dataset/
├── train/
│   ├── real/
│   ├── fake/
├── val/
│   ├── real/
│   ├── fake/
├── test/
│   ├── real/
│   ├── fake/
my_model.h5

Model Architecture

Input: 160x160x5 (RGB + Magnitude + Phase).
Layers: 5 Conv2D blocks (32→512 filters), BatchNorm, LeakyReLU, MaxPooling, Dropout (0.3-0.5), Dense (512), Dense (1, sigmoid).
Optimizer: Adam (lr=1e-4).
Loss: Binary Crossentropy.
Metrics: Accuracy, Precision, Recall, AUC.

Results

Test Accuracy: ~92% (varies by run).
AUC: ~0.98.

              precision    recall  f1-score   support

        real       0.93      0.94      0.93      3150
        fake       0.94      0.93      0.93      3150

    accuracy                           0.93      6300
   macro avg       0.93      0.93      0.93      6300
weighted avg       0.93      0.93      0.93      6300

Limitations

Requires GPU for efficient training.
Uses dataset subset; full dataset may improve performance.
Frequency preprocessing increases computational load.
No real-time inference pipeline (extendable via Flask).

Troubleshooting

Kaggle API: Verify kaggle.json placement and permissions.
Memory Errors: Reduce batch size or dataset size.
Model Loading: Ensure LeakyReLU custom object is defined.

Future Work

Train on full dataset.
Add real-time inference API.
Explore additional frequency features.

License
This project is licensed under the MIT License. See the LICENSE file for details.
Acknowledgments

Dataset: 140k Real and Fake Faces by xhlulu.
Built with TensorFlow/Keras and Google Colab.

Contact
For questions, contact : muhammadtouseefmt1@gmail.com
