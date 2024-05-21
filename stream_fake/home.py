import streamlit as st
import mysql.connector
import re
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2

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
    st.error("Database connection failed.")

# Load the deepfake detection model
deepfake_model_path = "C:\\Users\\Paras Sharma\\OneDrive\\Documents\\Deepfake\\model_15_64 (1).h5"
deepfake_model = tf.keras.models.load_model(deepfake_model_path)

def validate_name(name):
    if re.match(r"^[a-zA-Z]+\s[a-zA-Z]+$", name):
        return True
    else:
        st.warning("Please enter a valid name (e.g., Firstname Lastname).")
        return False

def validate_phone(phone):
    if re.match(r"^[0-9]{10}$", phone):
        return True
    else:
        st.warning("Please enter a valid 10-digit phone number.")
        return False

def validate_email(email):
    email_pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
    if re.match(email_pattern, email):
        return True
    else:
        st.warning("Please enter a valid email.")
        return False

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
    else:
        return None

def show_home():
    st.header(' ')
    st.markdown("<h1 style='text-align: center; color: black;'>AuthentiTech: Leveraging Machine Learning to Combat Deepfake Detection</h1>", unsafe_allow_html=True)
    st.header(' ', divider="rainbow")
    st.header(' ')

    st.markdown("<p style='font-size: medium;'>Enter Your Details</p>", unsafe_allow_html=True)

    NAME = st.text_input('Name: ', st.session_state.get('name', ''))
    if not validate_name(NAME):
        return

    PHONE = st.text_input('Contact Number(+91): ', max_chars=10)
    PHONE = PHONE.strip()  # Remove any leading/trailing spaces
    if not validate_phone(PHONE):
        return

    GENDER = st.selectbox('Enter gender', ('F', 'M', 'other'))

    EMAIL = st.text_input('Email: ', st.session_state.get('EMAIL', ''))
    if not validate_email(EMAIL):
        return

    if st.button("Submit"):
        try:
            sql = "INSERT INTO user_details (NAME, PHONE, EMAIL, GENDER) VALUES (%s, %s, %s, %s)"
            val = (NAME, PHONE, EMAIL, GENDER)
            mycursor.execute(sql, val)
            mydb.commit()
            st.session_state['name'] = NAME
            st.session_state['EMAIL'] = EMAIL
            st.success("Details submitted successfully!")
        except mysql.connector.Error as err:
            st.error(f"Error: {err}")
            print(f"Error executing SQL: {err}")

    st.write("Upload your image (JPEG, JPG, PNG) here (max size: 15 KB):")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], accept_multiple_files=False, key="file_uploader")

    if uploaded_file is not None:
        file_details = {"FileName": uploaded_file.name, "FileType": uploaded_file.type, "FileSize": uploaded_file.size}
        st.write(file_details)
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("Detect Now"):
            prediction = predict_deepfake(image)
            if prediction < 0.5:
                st.write("Fake Image")
            else:
                st.write("Real Image")
    else:
        st.warning("Please upload an image.")

if __name__ == "__main__":
    show_home()
