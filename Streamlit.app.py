from flask import Flask, render_template, request, jsonify
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np
#import gemini_api as gem
import io
from googletrans import Translator

app = Flask(__name__)

model_path = 'model.h5'
model = None
predicted_class_label=""

class_labels = {
    0: 'Apple scab',
    1: 'Apple Black rot',
    2: 'Apple rust',
    3: 'Apple healthy',
    4: 'Blueberry healthy',
    5: 'Cherry Powdery mildew',
    6: 'Cherry healthy',
    7: 'Corn Cercospora spot',
    8: 'Corn Common rust',
    9: 'Corn Leaf Blight',
    10: 'Corn healthy',
    11: 'Grape Black rot',
    12: 'Grape Esca(Black_Measles)',
    13: 'Grape Isariopsis Leaf Spot',
    14: 'Grape healthy',
    15: 'Orange Citrus greening',
    16: 'Peach Bacterial spot',
    17: 'Peach healthy',
    18: 'Pepper bell Bacterial spot',
    19: 'Pepper bell healthy',
    20: 'Potato Early blight',
    21: 'Potato Late blight',
    22: 'Potato healthy',
    23: 'Raspberry healthy',
    24: 'Soybean healthy',
    25: 'Squash Powdery mildew',
    26: 'Strawberry Leaf scorch',
    27: 'Strawberry healthy',
    28: 'Tomato Bacterial spot',
    29: 'Tomato Early blight',
    30: 'Tomato Late blight',
    31: 'Tomato Leaf Mold',
    32: 'Tomato Septoria leaf spot',
    33: 'Tomato Spider mites ',
    34: 'Tomato Target Spot',
    35: 'Tomato Yellow Leaf Curl Virus',
    36: 'Tomato mosaic virus',
    37: 'Tomato healthy'
}

def load_model_if_necessary():
    global model
    if model is None:
        model = load_model(model_path)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        load_model_if_necessary()
        if 'image' not in request.files:
            return 'No file part'
        
        file = request.files['image']

        if file.filename == '':
            return 'No selected file'

        try:
            img = image.load_img(io.BytesIO(file.read()), target_size=(100, 100))  
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.0
    
            predictions = model.predict(img_array)
            global predicted_class_label
            predicted_class_index = np.argmax(predictions)
            predicted_class_label = class_labels[predicted_class_index]

            return render_template('result.html', predicted_class=predicted_class_label)
        except Exception as e:
            return str(e)

@app.route('/translate', methods=['POST'])
def translate_text():
    data = request.get_json()
    text = data.get('text', '')
    
    translator = Translator()
    translated_text = translator.translate(text, src='en', dest='ta').text
    
    return jsonify({'translated_text': translated_text})

if __name__ == '__main__':
    app.run(debug=True)
            



