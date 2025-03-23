from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import os

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "static/uploads"

# Load Models
models = {
    "pneumonia": load_model("models/model1.h5"),
    "breast_cancer": load_model("models/model2.h5"),
    "lung_cancer": load_model("models/model3.h5"),
    "skin_cancer": load_model("models/model4.h5")
}

@app.route("/")
def index():
    return render_template("index.html")






# Load Pneumonia Model
pneumonia_model = tf.keras.models.load_model("models/model1.h5")
# Class labels for prediction output
CLASS_NAMES1 = ["Normal", "Pneumonia"]
# Function to preprocess image

def preprocess_pneumonia_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # Load as RGB (3 channels)
    image = cv2.resize(image, (128, 128))  # Resize
    image = image / 255.0  # Normalize (Convert pixel values 0-255 to 0-1)
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

@app.route("/pneumonia", methods=["GET", "POST"])
def pneumonia():
    if request.method == "POST":
        if "file" not in request.files:
            return render_template("1_pneumonia.html", error="No file uploaded.")

        file = request.files["file"]
        if file.filename == "":
            return render_template("1_pneumonia.html", error="No selected file.")

        # Save uploaded image
        image_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(image_path)

        # Preprocess image
        image = preprocess_pneumonia_image(image_path)

        # Predict class
        prediction = pneumonia_model.predict(image)
        predicted_class = CLASS_NAMES1[np.argmax(prediction)]

        return render_template("1_pneumonia.html", prediction=predicted_class, image_url=image_path)

    return render_template("1_pneumonia.html", prediction=None)






# Load breast_cancer Model
breast_cancer_model = tf.keras.models.load_model("models/model2.h5")
# Class labels for prediction output
CLASS_NAMES2 = ["Normal", "Benign", "Malignant"]
# Function to preprocess image


@app.route("/breast_cancer", methods=["GET", "POST"])
def breast_cancer():
    if request.method == "POST":
        if "file" not in request.files:
            return render_template("2_breast_cancer.html", error="No file uploaded.")

        file = request.files["file"]
        if file.filename == "":
            return render_template("2_breast_cancer.html", error="No selected file.")

        # Save uploaded image
        image_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(image_path)

        # Preprocess image
        image = preprocess_pneumonia_image(image_path)

        # Predict class
        prediction = breast_cancer_model.predict(image)
        predicted_class = CLASS_NAMES2[np.argmax(prediction)]

        return render_template("2_breast_cancer.html", prediction=predicted_class, image_url=image_path)

    return render_template("2_breast_cancer.html", prediction=None)







# Load lung_cancer Model
lung_cancer_model = tf.keras.models.load_model("models/model3.h5")
# Class labels for prediction output
CLASS_NAMES3 = ["Normal", "Benign", "Malignant"]
# Function to preprocess image

def preprocess_lung_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # Read image
    image = cv2.resize(image, (150, 150))  # Resize to (150, 150)
    image = image / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image


@app.route("/lung_cancer", methods=["GET", "POST"])
def lung_cancer():
    if request.method == "POST":
        if "file" not in request.files:
            return render_template("3_lung_cancer.html", error="No file uploaded.")

        file = request.files["file"]
        if file.filename == "":
            return render_template("3_lung_cancer.html", error="No selected file.")

        # Save uploaded image
        image_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(image_path)

        # Preprocess image
        image = preprocess_lung_image(image_path)
        print("Image Shape After Preprocessing:", image.shape)  # Debugging

        # Predict class
        prediction = lung_cancer_model.predict(image)
        predicted_class = CLASS_NAMES3[np.argmax(prediction)]

        return render_template("3_lung_cancer.html", prediction=predicted_class, image_url=image_path)

    return render_template("3_lung_cancer.html", prediction=None)





# Load Skin Cancer Model
skin_cancer_model = tf.keras.models.load_model("models/model4.h5")

# Define class names for the 7 categories
CLASS_NAMES4 = ["Actinic keratoses (akiec)", "Basal cell carcinoma (bcc)", "Benign keratosis-like (bkl)", 
                "Dermatofibroma (df)", "Melanoma (mel)", "Melanocytic nevi (nv)", "Vascular lesions (vasc)"]

# Function to preprocess the uploaded image for the model
def preprocess_skin_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # Load the image in color
    image = cv2.resize(image, (64,64))  # Resize to 128x128 (adjust as per model)
    image = image / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

@app.route("/skin_cancer", methods=["GET", "POST"])
def skin_cancer():
    if request.method == "POST":
        if "file" not in request.files:
            return render_template("4_skin_cancer.html", error="No file uploaded.")

        file = request.files["file"]
        if file.filename == "":
            return render_template("4_skin_cancer.html", error="No selected file.")

        # Save uploaded image
        image_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(image_path)

        # Preprocess image
        image = preprocess_skin_image(image_path)

        # Predict class
        prediction = skin_cancer_model.predict(image)
        predicted_class = CLASS_NAMES4[np.argmax(prediction)]  # Get class name

        return render_template("4_skin_cancer.html", prediction=predicted_class, image_url=image_path)

    return render_template("4_skin_cancer.html", prediction=None)






def process_request(model_name, template):
    if request.method == "POST":
        try:
            features = [float(request.form[f"feature{i}"]) for i in range(1, 3)]  # Adjust for your model
            input_data = np.array(features).reshape(1, -1)
            prediction = models[model_name].predict(input_data)[0][0]
            return render_template(template, prediction=round(prediction, 2))
        except Exception as e:
            return render_template(template, error=str(e))

    return render_template(template, prediction=None)

if __name__ == "__main__":
    app.run(debug=True)










