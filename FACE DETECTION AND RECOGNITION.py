import os
import cv2
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import font
from PIL import Image, ImageTk
import numpy as np
import face_recognition
import dlib

# Initialize the pre-trained models for face and landmark detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Path to directory containing known face images
known_faces_dir = 'known_faces'

# Ensure the directory exists
if not os.path.exists(known_faces_dir):
    raise FileNotFoundError(f"The directory {known_faces_dir} does not exist. Please create it and add known face images.")

# Load known face images and encode them
def load_known_faces(known_faces_dir):
    known_face_encodings = []
    known_face_names = []
    
    for filename in os.listdir(known_faces_dir):
        if filename.endswith(('.jpg', '.png')):
            image_path = os.path.join(known_faces_dir, filename)
            image = face_recognition.load_image_file(image_path)
            encoding = face_recognition.face_encodings(image)
            if encoding:
                known_face_encodings.append(encoding[0])
                known_face_names.append(filename.split('.')[0])
    
    return known_face_encodings, known_face_names

known_face_encodings, known_face_names = load_known_faces(known_faces_dir)

def detect_landmarks(image, face_location):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    (top, right, bottom, left) = face_location
    rect = dlib.rectangle(left, top, right, bottom)
    landmarks = predictor(gray, rect)

    # Draw landmarks (eyes, nose, mouth, etc.)
    for i in range(0, 68):
        x = landmarks.part(i).x
        y = landmarks.part(i).y
        cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

    return image

def recognize_faces(image):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_image)
    face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
    face_names = []

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]
        face_names.append(name)

    for (top, right, bottom, left), name in zip(face_locations, face_names):
        cv2.rectangle(image, (left, top), (right, bottom), (255, 0, 0), 2)
        cv2.putText(image, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

        # Detect and draw landmarks
        image = detect_landmarks(image, (top, right, bottom, left))

    message = "The detected person is registered" if "Unknown" not in face_names else "Error: The detected person is not the same"
    return image, message

def open_camera():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detected_frame, message = recognize_faces(frame)
        cv2.imshow('Camera Feed', detected_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            messagebox.showinfo("Result", message)
            break

def detect_from_file(file_path):
    image = cv2.imread(file_path)
    
    if image is None:
        messagebox.showerror("Error", "Failed to load the image. Please check the file path or try another image.")
        return None, "Image not loaded"
    
    detected_image, message = recognize_faces(image)
    result_path = 'result_with_landmarks.jpg'
    cv2.imwrite(result_path, detected_image)
    return result_path, message


def upload_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        result_path, message = detect_from_file(file_path)
        load_image(result_path)
        messagebox.showinfo("Result", message)

def load_image(image_path):
    image = Image.open(image_path)
    image = ImageTk.PhotoImage(image)
    label_image.config(image=image)
    label_image.image = image

def save_image():
    file_path = filedialog.asksaveasfilename(defaultextension=".jpg", filetypes=[("JPEG files", "*.jpg")])
    if file_path:
        result_path = 'result_with_landmarks.jpg'
        img = Image.open(result_path)
        img.save(file_path)
        messagebox.showinfo("Info", "Image saved successfully!")

def exit_app():
    root.destroy()

# Create the main window
root = tk.Tk()
root.title('Face and Landmark Detection Application')
root.geometry('800x600')
root.configure(bg='lightgrey')

# Custom fonts
title_font = font.Font(family="Helvetica", size=16, weight="bold")
button_font = font.Font(family="Helvetica", size=12, weight="bold")

# Create and place widgets
title_label = tk.Label(root, text='Face and Landmark Detection Application', font=title_font, bg='lightgrey')
title_label.pack(pady=10)

btn_upload = tk.Button(root, text='Upload Image', command=upload_image, font=button_font, bg='lightblue', width=20, height=2)
btn_upload.pack(pady=5)

btn_camera = tk.Button(root, text='Open Camera', command=open_camera, font=button_font, bg='lightgreen', width=20, height=2)
btn_camera.pack(pady=5)

btn_save = tk.Button(root, text='Save Image', command=save_image, font=button_font, bg='lightcoral', width=20, height=2)
btn_save.pack(pady=5)

btn_exit = tk.Button(root, text='Exit', command=exit_app, font=button_font, bg='lightpink', width=20, height=2)
btn_exit.pack(pady=5)

label_image = tk.Label(root, bg='lightgrey')
label_image.pack(pady=10, expand=True)

# Run the application
root.mainloop()
