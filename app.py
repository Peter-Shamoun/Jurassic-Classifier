from flask import Flask, render_template, request, jsonify, logging
from flask_wtf.csrf import CSRFProtect
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_swagger_ui import get_swaggerui_blueprint
from dino import model_prediction, get_model_info
import base64
import logging
import os
import time

# Configure application
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'jurassic-classifier-dev-key')

# Set up CSRF protection
csrf = CSRFProtect(app)

# Set up rate limiting
limiter = Limiter(
    key_func=get_remote_address,
    app=app,
    default_limits=["200 per day", "50 per hour"]
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Swagger configuration
SWAGGER_URL = '/api/docs'
API_URL = '/static/swagger.json'

swaggerui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config={
        'app_name': "Jurassic Classifier API"
    }
)
app.register_blueprint(swaggerui_blueprint, url_prefix=SWAGGER_URL)

@app.route('/')
def home():
    """Render the home page with the dinosaur classification form"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
@limiter.limit("10 per minute")
@csrf.exempt  # Exempt the API endpoint from CSRF (typically handled by frontend token)
def predict():
    """Process image data and return dinosaur classification prediction"""
    start_time = time.time()
    
    try:
        # Validate request data
        if not request.json or 'image' not in request.json:
            app.logger.warning("Invalid request: missing image data")
            return jsonify({'error': 'No image data provided'}), 400
            
        image_data = request.json['image']
        
        # Validate image format
        if not image_data.startswith('data:image/'):
            app.logger.warning("Invalid request: incorrect image format")
            return jsonify({'error': 'Invalid image format'}), 400
        
        # Remove the data URL prefix to get raw base64
        try:
            image_data = image_data.split(',')[1]
        except IndexError:
            app.logger.warning("Invalid request: malformed image data")
            return jsonify({'error': 'Malformed image data'}), 400
            
        # Convert base64 to bytes
        try:
            image_bytes = base64.b64decode(image_data)
        except Exception as e:
            app.logger.error(f"Base64 decoding error: {str(e)}")
            return jsonify({'error': 'Invalid base64 encoding'}), 400
        
        # Get prediction with timing
        app.logger.info(f"Processing prediction request")
        prediction = model_prediction(image_bytes)
        
        # Log successful prediction
        processing_time = time.time() - start_time
        app.logger.info(f"Prediction completed in {processing_time:.2f}s")
        
        return jsonify(prediction)
        
    except Exception as e:
        app.logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': 'Server error during prediction'}), 500

@app.route('/api/model-info', methods=['GET'])
def model_info():
    """Return information about the model"""
    try:
        info = get_model_info()
        return jsonify(info)
    except Exception as e:
        app.logger.error(f"Error retrieving model info: {str(e)}")
        return jsonify({'error': 'Error retrieving model information'}), 500

if __name__ == '__main__':
    app.run(debug=True) 