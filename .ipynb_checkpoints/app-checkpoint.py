from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import cv2
from skimage.feature import local_binary_pattern

app = Flask(__name__)

model = joblib.load('C:/Users/Iman/Machine_Learning_Project/tomato_disease_model.pkl')
label_encoder = joblib.load('C:/Users/Iman/Machine_Learning_Project/label_encoder.pkl')


def preprocess_image(image, size=(128, 128)):
    img = cv2.resize(image, size)
    img = (img * 255).astype(np.uint8)  # Normalize to [0, 1], then scale to [0, 255] and downcast to uint8
    return img

def extract_color_histogram(image, bins=(8, 8, 8)):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

def extract_lbp(image, P=8, R=1):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray_image, P, R, method="uniform")
    (hist, _) = np.histogram(lbp.ravel(),
                             bins=np.arange(0, P + 3),
                             range=(0, P + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)
    return hist

def extract_features(image):
    color_hist = extract_color_histogram(image)
    lbp_hist = extract_lbp(image)
    return np.hstack((color_hist, lbp_hist))

@app.route('/')
def home():
    return render_template('C:/Users/Iman/Machine_Learning_Project/index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    file.save('uploaded_image.jpg')
    image = cv2.imread('uploaded_image.jpg')
    preprocessed_image = preprocess_image(image)
    features = extract_features(preprocessed_image).reshape(1, -1)
    prediction = model.predict(features)
    label = label_encoder.inverse_transform(prediction)[0]

    return jsonify({'label': label})

if __name__ == '__main__':
    app.run(debug=True)
