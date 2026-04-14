from flask import Flask, request, jsonify
from flask_cors import CORS
from inference_sdk import InferenceHTTPClient
import tempfile
import os
app = Flask(__name__)
CORS(app)

# Roboflow client
CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="21PhztC7wjwP0VK8YIt7"
)

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['image']
    lat = request.form.get('lat')
    lng = request.form.get('lng')

    # Save image temporarily
    temp = tempfile.NamedTemporaryFile(delete=False)
    file.save(temp.name)

    # Run model
    result = CLIENT.infer(temp.name, model_id="pothole-voxrl/1")

    predictions = result.get("predictions", [])

    pothole_detected = len(predictions) > 0

    confidence = predictions[0]["confidence"] if pothole_detected else 0

    return jsonify({
        "pothole": pothole_detected,
        "confidence": confidence,
        "lat": float(lat),
        "lng": float(lng)
    })



if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
