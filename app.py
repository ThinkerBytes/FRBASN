from flask import Flask, render_template, redirect, url_for
from datetime import datetime
from keras.models import load_model
import cv2
import numpy as np
import os
import threading

app = Flask(__name__)

# Load the Teachable Machine model and labels
model = load_model("keras_Model.h5", compile=False)
class_names = open("labels.txt", "r").readlines()

# Load the Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize a dictionary to keep track of attendance
attendance_dict = {name.strip(): False for name in class_names}

# Initialize camera
camera = cv2.VideoCapture(0)

# Global variable to control attendance detection
is_detecting_attendance = False

# Global variable to store the path of the current attendance file
current_attendance_file = 'attendance/attendance.csv'

# Function to detect and mark attendance
def mark_attendance():
    global is_detecting_attendance
    
    while is_detecting_attendance:
        ret, frame = camera.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            face = cv2.resize(face, (224, 224), interpolation=cv2.INTER_AREA)
            face = np.asarray(face, dtype=np.float32).reshape(1, 224, 224, 3)
            face = (face / 127.5) - 1
            prediction = model.predict(face)
            index = np.argmax(prediction)
            class_name = class_names[index]
            confidence_score = prediction[0][index]

            if confidence_score > 0.9 and not attendance_dict[class_name.strip()]:
                attendance_dict[class_name.strip()] = True
                # Add attendance to CSV file
                with open(current_attendance_file, 'a') as f:
                    f.write(f'{class_name.strip()},{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        cv2.imshow("Webcam Image", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

# Route for home page
@app.route('/')
def home():
    # Create the 'attendance' folder if it doesn't exist
    if not os.path.exists('attendance'):
        os.makedirs('attendance')

    # Create the 'attendance.csv' file if it doesn't exist
    if not os.path.exists(current_attendance_file):
        with open(current_attendance_file, 'w') as f:
            f.write('Name,Time\n')

    # Read the latest attendance data
    attendance = []
    with open(current_attendance_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split(',')
            attendance.append({'Name': parts[0], 'Time': parts[1]})
    
    # Pass the latest attendance data to the template
    return render_template('index.html', latest_attendance=attendance)


# Route for taking attendance
@app.route('/take_attendance')
def take_attendance():
    global is_detecting_attendance
    is_detecting_attendance = True
    threading.Thread(target=mark_attendance).start()  # Start attendance detection in a separate thread
    return redirect(url_for('attendance_list'))  # Redirect to the attendance list while attendance detection is ongoing


# Route for stopping attendance detection
@app.route('/stop_attendance')
def stop_attendance():
    global is_detecting_attendance
    is_detecting_attendance = False
    return redirect(url_for('home'))

# Route for showing current attendance list
@app.route('/attendance_list')
def attendance_list():
    # Read attendance from CSV file
    attendance = []
    with open(current_attendance_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split(',')
            attendance.append({'Name': parts[0], 'Time': parts[1]})
    return render_template('attendance_list.html', attendance=attendance)

# Route for creating a new period (new attendance file)
@app.route('/new_period')
def new_period():
    global current_attendance_file
    # Increment the period number
    period_number = len([file for file in os.listdir('attendance') if file.startswith('attendance-')])

    # Create a new attendance file for the new period
    new_attendance_file = f'attendance/attendance-{period_number + 1}-{datetime.now().strftime("%m_%d_%y")}.csv'
    with open(new_attendance_file, 'w') as f:
        f.write('Name,Time\n')

    # Update the current attendance file path
    current_attendance_file = new_attendance_file

    # Redirect to the home page
    return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True)
