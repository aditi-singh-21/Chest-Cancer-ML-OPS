from flask import Flask, request, jsonify, send_from_directory, render_template
import os
from flask_cors import CORS, cross_origin
from chest_cancer_classifier.utils.common import decodeImage
from chest_cancer_classifier.pipeline.predictions import PredicationPipeline

os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)

class ClientApp:
    def __init__(self):
        self.filename = "inputImage.jpg"
        self.classifier = PredicationPipeline(self.filename)

@app.route("/", methods=['GET'])
@cross_origin()
def home():
    return render_template('index.html')  # Serve the index.html from the templates folder

@app.route("/train", methods=["GET", "POST"])
@cross_origin()
def trainRoute():
    os.system("python main.py")
    return "Training done successfully"

@app.route("/predict", methods=['POST'])
@cross_origin()
def predict():
    if request.method == 'POST':
        # Assuming the image is sent as base64 encoded string in JSON
        data = request.json
        image_data = data.get('image')

        # Decode the image and save it
        decodeImage(image_data, "inputImage.jpg")

        # Run the classifier
        client_app = ClientApp()
        result = client_app.classifier.predict()  # Adjust this method call according to your pipeline

        # Return the prediction result as JSON
        return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)
