Leaf Disease Prediction

Overview

This project is a web-based application that predicts plant leaf diseases using a deep learning model. The aim is to identify whether a leaf is healthy or affected by a disease by uploading an image. It is built as part of learning how machine learning and computer vision can be applied to real-world problems.

What this project does

The application allows users to upload an image of a plant leaf. The image is processed and passed to a trained model, which predicts the disease or identifies if the leaf is healthy. The result is then displayed on a web page

Tech Stack

Python – main programming language

TensorFlow / Keras – model building and training

OpenCV and NumPy – image processing

Flask – backend framework

HTML and CSS – frontend

How to run the project

Step 1: Clone the repository

git clone https://github.com/Dhruvika1208/Leaf_disease_prediction.git

cd Leaf_disease_prediction

Step 2: Install dependencies

pip install -r requirements.txt

Step 3: Run the application

python app.py

Step 4: Open in browser

http://127.0.0.1:5000

How it works

The model is trained on a dataset of plant leaf images where each image belongs to a specific class such as a disease type or a healthy leaf.

When a user uploads an image:

The image is resized and preprocessed

It is passed into the trained model

The model predicts the class

The result is displayed on the interface


Project Structure

Leaf_disease_prediction/

│── static/

│── templates/

│── model/

│── app.py

│── train.py

│── requirements.txt

│── README.md

Results

The model is able to classify leaf images into different categories with reasonable accuracy depending on the dataset and training.

Future Improvements

Improve accuracy by using more training data

Add more disease categories

Provide remedies or suggestions for detected diseases

Deploy the project online
Extend to mobile-based detection
