import Flask
import render_template
import request
from keras.models import load_model
from PIL import Image
import numpy as np
import pickle
import io
import base64

app = Flask(__name__)

# Load your trained model
# model = pickle.load(open('model.pkl', 'rb'))
model = load_model('model_catsVSdogs_10epoch.h5')


@app.route('/')
def index():
    return render_template('index.html', uploaded_image=None, prediction=None, confidence=None, error=None)


@app.route('/predict', methods=['POST'])
def predict():
    uploaded_file = request.files['file']
    if uploaded_file.filename != '':
        # Preprocess the image
        image = Image.open(uploaded_file)
        image = image.resize((128, 128))
        image_array = np.asarray(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)

        # Make prediction
        prediction = model.predict(image_array)
        predicted_class = np.argmax(prediction)
        class_labels = {0: 'Cat', 1: 'Dog'}
        predicted_label = class_labels[predicted_class]
        confidence = prediction[0][predicted_class]

        # Convert the image to base64 for display in HTML
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        img_str = "data:image/jpeg;base64," + \
            base64.b64encode(buffered.getvalue()).decode("utf-8")

        return render_template('index.html', uploaded_image=img_str, prediction=predicted_label, confidence=confidence, error=None)
    else:
        return render_template('index.html', uploaded_image=None, prediction=None, confidence=None, error='Please upload an image.')


if __name__ == '__main__':
    app.run(debug=True)
