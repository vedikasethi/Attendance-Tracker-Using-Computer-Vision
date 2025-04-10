import cv2
import numpy as np
from mtcnn import MTCNN
from keras_facenet import FaceNet
import tkinter as tk
from tkinter import messagebox
import time
import threading
import sys
from datetime import datetime
import faiss
import mysql.connector

# Database config
DB_CONFIG = {
    'host': 'localhost',
    'user': 'user123',
    'password': 'root',
    'database': 'attendance'
}

# Initialize MTCNN and FaceNet
detector = MTCNN()
embedder = FaceNet()
attendance_marked = set()
students_within_one_hour = []

# Load embeddings from MySQL
def load_embeddings():
    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor()
    cursor.execute("SELECT name, sap_id, embedding FROM face_embeddings")
    records = cursor.fetchall()
    conn.close()

    embeddings = []
    names = []
    for name, sap_id, embedding_str in records:
        embedding = np.array(list(map(float, embedding_str.split(','))))
        embeddings.append(embedding)
        names.append(f"{name}_{sap_id}")
    
    embeddings = np.array(embeddings).astype('float32')
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, names

# Find closest match
def find_closest_match(embedding, faiss_index, names):
    embedding = np.expand_dims(embedding, axis=0).astype('float32')
    distances, indices = faiss_index.search(embedding, 1)
    return names[indices[0][0]] if distances[0][0] < 0.6 else "Unknown"

# Save new face to MySQL
def capture_and_save_embeddings(name, sap_id):
    cap = cv2.VideoCapture(0)
    start_time = time.time()
    captured = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        faces = detector.detect_faces(frame)
        if faces:
            for face in faces:
                x1, y1, width, height = face['box']
                x1, y1 = abs(x1), abs(y1)
                x2, y2 = x1 + width, y1 + height
                face_img = frame[y1:y2, x1:x2]
                face_img = cv2.resize(face_img, (160, 160))
                face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                if time.time() - start_time > 10 and not captured:
                    face_embedding = embedder.embeddings([face_img])[0]
                    embedding_str = ",".join(map(str, face_embedding))
                    conn = mysql.connector.connect(**DB_CONFIG)
                    cursor = conn.cursor()
                    cursor.execute("INSERT INTO face_embeddings (name, sap_id, embedding) VALUES (%s, %s, %s)", (name, sap_id, embedding_str))
                    conn.commit()
                    conn.close()
                    captured = True
                    messagebox.showinfo("Success", f"Face data for {name} saved.")
                    break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

# Start recognition and mark attendance
def start_face_recognition():
    faiss_index, names = load_embeddings()
    cap = cv2.VideoCapture(0)
    for i in range(3, 0, -1):
        messagebox.showinfo("Countdown", f"Starting in {i}...")
        time.sleep(1)

    first_attendance_timestamp = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        faces = detector.detect_faces(frame)
        if faces:
            face_images = []
            coords = []
            for face in faces:
                x1, y1, width, height = face['box']
                x1, y1 = abs(x1), abs(y1)
                x2, y2 = x1 + width, y1 + height
                face_img = frame[y1:y2, x1:x2]
                face_img = cv2.resize(face_img, (160, 160))
                face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                face_images.append(face_img)
                coords.append((x1, y1, x2, y2))
            embeddings = embedder.embeddings(face_images)

            for i, embedding in enumerate(embeddings):
                name = find_closest_match(embedding, faiss_index, names)
                x1, y1, x2, y2 = coords[i]

                if name != "Unknown" and name not in attendance_marked:
                    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    name_split = name.split('_')
                    conn = mysql.connector.connect(**DB_CONFIG)
                    cursor = conn.cursor()
                    cursor.execute("INSERT INTO attendance (name, sap_id, timestamp) VALUES (%s, %s, %s)",
                                   (name_split[0], name_split[1], timestamp))
                    conn.commit()
                    conn.close()
                    attendance_marked.add(name)

                    if first_attendance_timestamp is None:
                        first_attendance_timestamp = datetime.now()
                    attendance_time = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
                    if (attendance_time - first_attendance_timestamp).seconds <= 3600:
                        students_within_one_hour.append({
                            "name": name,
                            "timestamp": timestamp
                        })

                    messagebox.showinfo("Attendance Marked", f"Attendance marked for {name} at {timestamp}.")
                elif name == "Unknown":
                    messagebox.showinfo("Unknown Face", "Face not recognized.")

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, name, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if students_within_one_hour:
        details_str = "\n".join([f"{s['name']} at {s['timestamp']}" for s in students_within_one_hour])
        messagebox.showinfo("Students Within 1 Hour", f"{details_str}")
    else:
        messagebox.showinfo("No Students", "No students arrived within 1 hour.")
    cap.release()
    cv2.destroyAllWindows()

# GUI Setup
def start_gui():
    root = tk.Tk()
    root.title("Face Recognition Attendance System")

    tk.Label(root, text="Name").grid(row=0, column=0)
    name_entry = tk.Entry(root)
    name_entry.grid(row=0, column=1)

    tk.Label(root, text="SAP ID").grid(row=1, column=0)
    sap_id_entry = tk.Entry(root)
    sap_id_entry.grid(row=1, column=1)

    def on_capture():
        name = name_entry.get()
        sap_id = sap_id_entry.get()
        if name and sap_id:
            threading.Thread(target=capture_and_save_embeddings, args=(name, sap_id)).start()
        else:
            messagebox.showerror("Error", "Enter both name and SAP ID.")

    def on_recognize():
        threading.Thread(target=start_face_recognition).start()

    tk.Button(root, text="Capture Face", command=on_capture).grid(row=2, column=0)
    tk.Button(root, text="Start Recognition", command=on_recognize).grid(row=2, column=1)

    root.mainloop()

# Start app
start_gui()
