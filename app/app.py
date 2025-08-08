from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

# Load the pre-trained model
model = joblib.load('outputs/best_model/model.pkl')

# Initialize Flask app
app = Flask(__name__, template_folder='template')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from the request
        data = request.get_json(force=True)
        
        # Extract features from the JSON data
        features = np.array(data['features']).reshape(1, -1)
        
        # Make prediction using the loaded model
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0].tolist()
        
        # Return the prediction as a JSON response
        return jsonify({'prediction': int(prediction), 'probability': probability}), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400
    
@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, port=5050)