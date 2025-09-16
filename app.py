# from flask import Flask, request, jsonify, render_template
# import os
# from flask_cors import CORS, cross_origin
# from cnnClassifier.utils.common import decodeImage
# from cnnClassifier.pipeline.prediction import PredictionPipeline



# os.putenv('LANG', 'en_US.UTF-8')
# os.putenv('LC_ALL', 'en_US.UTF-8')

# app = Flask(__name__)
# CORS(app)


# class ClientApp:
#     def __init__(self):
#         self.filename = "inputImage.jpg"
#         self.classifier = PredictionPipeline(self.filename)


# @app.route("/", methods=['GET'])
# @cross_origin()
# def home():
#     return render_template('index.html')




# @app.route("/train", methods=['GET','POST'])
# @cross_origin()
# def trainRoute():
#     os.system("python main.py")
#     # os.system("dvc repro")
#     return "Training done successfully!"



# @app.route("/predict", methods=['POST'])
# @cross_origin()
# def predictRoute():
#     image = request.json['image']
#     decodeImage(image, clApp.filename)
#     result = clApp.classifier.predict()
#     return jsonify(result)


# if __name__ == "__main__":
#     clApp = ClientApp()

#     app.run(host='0.0.0.0', port=8080) #for AWS

import gradio as gr
from cnnClassifier.pipeline.prediction import PredictionPipeline
from cnnClassifier.utils.common import decodeImage
import numpy as np
from PIL import Image

# Prediction function
def predict(img: Image.Image):
    filename = "inputImage.jpg"
    img.save(filename)
    pipeline = PredictionPipeline(filename)
    result = pipeline.predict()
    return result

# Gradio Interface
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload Kidney MRI Image"),
    outputs="json",
    title="Kidney Disease Classification",
    description="Upload an MRI scan and classify as Tumor or Normal."
)

if __name__ == "__main__":
    demo.launch()
