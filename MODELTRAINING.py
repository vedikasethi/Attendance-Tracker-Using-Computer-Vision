import cv2
import numpy as np
from mtcnn import MTCNN
from keras_facenet import FaceNet
import tkinter as tk
from tkinter import messagebox
import csv
import time
import threading
import sys
from datetime import datetime
import faiss  # Fast Approximate Nearest Neighbor search

# Initialize MTCNN for face detection
detector = MTCNN()

# Load the pre-trained FaceNet model
embedder = FaceNet()

# Global set to track attendance for the session
attendance_marked = set()

# List to store students' details who arrived within 1 hour
students_within_one_hour = []

# Load saved embeddings and names, and use FAISS for fast nearest neighbor search
def load_embeddings():
    embeddings = []
    names = []
    with open('face_embeddings.csv', 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            name = row[0]
            embedding = np.array(row[1:], dtype=float)
            embeddings.append(embedding)
            names.append(name)
    
    # Convert list to numpy array
    embeddings = np.array(embeddings)
    
    # Use FAISS for fast nearest neighbor search
    index = faiss.IndexFlatL2(embeddings.shape[1])  # L2 distance metric
    index.add(embeddings)  # Add embeddings to FAISS index
    
    return index, names

# Function to find the closest match using FAISS
def find_closest_match(embedding, faiss_index, names):
    embedding = np.expand_dims(embedding, axis=0).astype('float32')  # FAISS requires float32
    distances, indices = faiss_index.search(embedding, 1)  # Search for the closest match
    
    # Adjust the threshold for recognition. Lower threshold to improve match rate.
    if distances[0][0] < 0.6:  # Use a more lenient threshold for matching
        return names[indices[0][0]]
    else:
        return "Unknown"

# Function to capture face and save embedding
def capture_and_save_embeddings(name, sap_id):
    # Open the webcam
    cap = cv2.VideoCapture(0)
    start_time = time.time()
    captured = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect faces
        faces = detector.detect_faces(frame)
        
        if faces:
            for face in faces:
                # Only consider the first detected face (most prominent)
                x1, y1, width, height = face['box']
                x1, y1 = abs(x1), abs(y1)
                x2, y2 = x1 + width, y1 + height
                
                # Extract the face
                face_img = frame[y1:y2, x1:x2]
                face_img = cv2.resize(face_img, (160, 160))
                face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                
                # Capture face for 5-10 seconds
                if time.time() - start_time > 10 and not captured:
                    face_embedding = embedder.embeddings([face_img])[0]
                    with open('face_embeddings.csv', mode='a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow([name + "_" + sap_id] + face_embedding.tolist())  # Use SAP ID instead of roll number
                    captured = True
                    messagebox.showinfo("Success", f"Face data for {name} with SAP ID {sap_id} saved.")
                    break
        
        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Function to handle cleanup and exit
def cleanup_and_exit(cap):
    cap.release()
    cv2.destroyAllWindows()
    sys.exit()

# Function to start face recognition with multiple faces in the frame
def start_face_recognition():
    faiss_index, names = load_embeddings()
    cap = cv2.VideoCapture(0)

    # Countdown before starting
    for i in range(3, 0, -1):
        messagebox.showinfo("Countdown", f"Starting in {i}...")
        time.sleep(1)

    first_attendance_timestamp = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect faces
        faces = detector.detect_faces(frame)
        if faces:
            face_images = []
            for face in faces:
                # Only process the first detected face (most prominent)
                x1, y1, width, height = face['box']
                x1, y1 = abs(x1), abs(y1)
                x2, y2 = x1 + width, y1 + height

                # Extract and preprocess the face
                face_img = frame[y1:y2, x1:x2]
                face_img = cv2.resize(face_img, (160, 160))
                face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

                face_images.append(face_img)

            # Generate embeddings for all detected faces in a batch
            embeddings = embedder.embeddings(face_images)

            for i, embedding in enumerate(embeddings):
                name = find_closest_match(embedding, faiss_index, names)

                # Check if this name's attendance has already been marked
                if name != "Unknown" and name not in attendance_marked:
                    # Mark attendance
                    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    with open('attendance_log.csv', mode='a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow([name, timestamp])  # Log name and timestamp for attendance

                    # Add to the attendance_marked set to avoid re-marking
                    attendance_marked.add(name)

                    # If it's the first student, store their timestamp
                    if first_attendance_timestamp is None:
                        first_attendance_timestamp = datetime.now()

                    # Check if the attendance is within 1 hour of the first attendance
                    attendance_time = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
                    if first_attendance_timestamp and (attendance_time - first_attendance_timestamp).seconds <= 3600:
                        # Add student to the list within 1 hour
                        student_details = {"name": name, "timestamp": timestamp}
                        students_within_one_hour.append(student_details)

                    messagebox.showinfo("Attendance Marked", f"Attendance marked for {name} at {timestamp}.")
                elif name == "Unknown":
                    messagebox.showinfo("Unknown Face", "Face not recognized.")

                # Draw bounding box and label with name and SAP ID
                x1, y1, width, height = faces[i]['box']
                cv2.rectangle(frame, (x1, y1), (x2, y1 + height), (0, 255, 0), 2)
                cv2.putText(frame, f"{name}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Display the frame
        cv2.imshow("Face Recognition", frame)

        # Break the loop after a set time or if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # After attendance is completed, show list of students within 1 hour and the total count
    if students_within_one_hour:
        details_str = "\n".join([f"Name: {student['name']}, Timestamp: {student['timestamp']}" for student in students_within_one_hour])
        messagebox.showinfo("Students Within 1 Hour", f"Total Students: {len(students_within_one_hour)}\n\n{details_str}")
    else:
        messagebox.showinfo("No Students Within 1 Hour", "No students arrived within 1 hour.")
    
    cleanup_and_exit(cap)

# GUI for taking input and starting face recognition
def start_gui():
    root = tk.Tk()
    root.title("Face Recognition Exam System")

    tk.Label(root, text="Enter Name:").grid(row=0, column=0)
    name_entry = tk.Entry(root)
    name_entry.grid(row=0, column=1)

    tk.Label(root, text="Enter SAP ID:").grid(row=1, column=0)  # Change label to SAP ID
    sap_id_entry = tk.Entry(root)  # Use SAP ID entry field
    sap_id_entry.grid(row=1, column=1)

    def on_start_capture():
        name = name_entry.get()
        sap_id = sap_id_entry.get()  # Get SAP ID instead of roll number
        if name and sap_id:
            threading.Thread(target=capture_and_save_embeddings, args=(name, sap_id)).start()
        else:
            messagebox.showerror("Error", "Please enter both name and SAP ID.")  # Update message

    def on_start_recognition():
        threading.Thread(target=start_face_recognition).start()

    tk.Button(root, text="Capture Face", command=on_start_capture).grid(row=2, column=0)
    tk.Button(root, text="Start Recognition", command=on_start_recognition).grid(row=2, column=1)

    root.mainloop()

# Run the GUI
start_gui()
