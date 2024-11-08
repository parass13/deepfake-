import os
import re
import mysql.connector
import tensorflow as tf
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, flash
from PIL import Image
import cv2

# Initialize Flask app
app = Flask(__name__)
app.secret_key = "your_secret_key"  # Required for flash messages

# Initialize database connection
try:
    mydb = mysql.connector.connect(
        host="localhost",
        user="root",
        password="12345",
        database="user_info"
    )
    mycursor = mydb.cursor()
    print("Connection Established")
except mysql.connector.Error as err:
    print(f"Error: {err}")
    flash("Database connection failed.")
    mydb = None

# Load the deepfake detection model
deepfake_model_path = "deepfake_model.h5"  # Adjust the path to your model
deepfake_model = tf.keras.models.load_model(deepfake_model_path)

# Utility functions for validation
def validate_name(name):
    return bool(re.match(r"^[a-zA-Z]+\s[a-zA-Z]+$", name))

def validate_phone(phone):
    return bool(re.match(r"^[0-9]{10}$", phone))

def validate_email(email):
    return bool(re.match(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", email))

def preprocess_image(image):
    try:
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        image = cv2.resize(image, (128, 128))  # Resize to match model input size
        image = image.astype(np.float32) / 255.0  # Normalize pixel values
        return np.expand_dims(image, axis=0)  # Add batch dimension
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

def predict_deepfake(image):
    preprocessed_image = preprocess_image(image)
    if preprocessed_image is not None:
        prediction = deepfake_model.predict(preprocessed_image)
        return prediction[0][0]  # Assuming the model outputs a single value between 0 and 1
    return None

# Routes for different pages
@app.route('/')
def home():
    return render_template("home.html")

@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/user_guide')
def user_guide():
    return render_template("user_guide.html")

@app.route('/submit_details', methods=["POST"])
def submit_details():
    if request.method == "POST":
        name = request.form["name"]
        phone = request.form["phone"]
        email = request.form["email"]
        gender = request.form["gender"]
        
        if not validate_name(name):
            flash("Please enter a valid name (Firstname Lastname).")
            return redirect(url_for('home'))
        
        if not validate_phone(phone):
            flash("Please enter a valid phone number.")
            return redirect(url_for('home'))
        
        if not validate_email(email):
            flash("Please enter a valid email.")
            return redirect(url_for('home'))
        
        if mydb:
            try:
                sql = "INSERT INTO user_details (NAME, PHONE, EMAIL, GENDER) VALUES (%s, %s, %s, %s)"
                val = (name, phone, email, gender)
                mycursor.execute(sql, val)
                mydb.commit()
                flash("Details submitted successfully!")
            except mysql.connector.Error as err:
                flash(f"Error: {err}")
        return redirect(url_for('home'))

@app.route('/upload_image', methods=["POST"])
def upload_image():
    if 'file' not in request.files:
        flash("No file uploaded.")
        return redirect(url_for('home'))
    
    uploaded_file = request.files['file']
    
    if uploaded_file.filename != '':
        image = Image.open(uploaded_file)
        prediction = predict_deepfake(image)
        
        result = "Real Image" if prediction >= 0.5 else "Fake Image"
        flash(f"Prediction: {result}")
        
    else:
        flash("Please upload an image.")
    
    return redirect(url_for('home'))

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
