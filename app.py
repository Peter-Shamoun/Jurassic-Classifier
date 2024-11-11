from flask import Flask, render_template, request, jsonify
from dino import model_prediction
import base64

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        image_data = request.json['image']
        # Remove the data URL prefix to get raw base64
        image_data = image_data.split(',')[1]
        # Convert base64 to bytes
        image_bytes = base64.b64decode(image_data)
        
        # Get prediction
        prediction = model_prediction(image_bytes)
        return jsonify({'prediction': prediction})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True) 