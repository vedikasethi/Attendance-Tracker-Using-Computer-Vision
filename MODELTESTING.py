import os
import cv2
import numpy as np
from keras_facenet import FaceNet
from mtcnn import MTCNN
import csv
from datetime import datetime
import faiss
import mysql.connector
import ast


# Initialize the MTCNN and FaceNet models
detector = MTCNN()
embedder = FaceNet()

# Function to load embeddings from the specified path
def load_embeddings():
    embeddings = []
    names = []

    # Connect to MySQL
    conn = mysql.connector.connect(
        host="localhost",
        user="user123",
        password="root",
        database="attendance"
    )
    cursor = conn.cursor()

    cursor.execute("SELECT name, embedding FROM face_embeddings")
    for name, embedding_str in cursor.fetchall():
        embedding = np.array(ast.literal_eval(embedding_str), dtype=float)
        names.append(name)
        embeddings.append(embedding)

    conn.close()

    embeddings = np.array(embeddings)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    return index, names


# Function to recognize faces and mark attendance
def start_face_recognition():
    faiss_index, names = load_embeddings()  # Load embeddings and FAISS index
    cap = cv2.VideoCapture(0)  # Start video capture from webcam

    attendance_marked = set()  # Set to track who has already marked attendance

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect faces in the frame using MTCNN
        faces = detector.detect_faces(frame)
        if faces:
            face_images = []
            for face in faces:
                x1, y1, width, height = face['box']
                x1, y1 = abs(x1), abs(y1)
                x2, y2 = x1 + width, y1 + height
                face_img = frame[y1:y2, x1:x2]
                face_img = cv2.resize(face_img, (160, 160))
                face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                face_images.append(face_img)

            embeddings = embedder.embeddings(face_images)

            for i, embedding in enumerate(embeddings):
                name = find_closest_match(embedding, faiss_index, names)

                if name != "Unknown" and name not in attendance_marked:
                    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                    # Insert attendance into MySQL
                    try:
                        conn = mysql.connector.connect(**MYSQL_CONFIG)
                        cursor = conn.cursor()
                        cursor.execute(
                            "INSERT INTO attendance (name, timestamp) VALUES (%s, %s)",
                            (name, timestamp)
                        )
                        conn.commit()
                        cursor.close()
                        conn.close()
                        print(f"Attendance marked for {name} at {timestamp}.")
                        attendance_marked.add(name)

                    except mysql.connector.Error as err:
                        print(f"MySQL Error: {err}")

                elif name == "Unknown":
                    print("Unknown face detected.")

                x1, y1, width, height = faces[i]['box']
                cv2.rectangle(frame, (x1, y1), (x2, y1 + height), (0, 255, 0), 2)
                cv2.putText(frame, f"{name}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow("Face Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Helper function to find the closest match for a face in the FAISS index
def find_closest_match(embedding, faiss_index, names):
    embedding = np.expand_dims(embedding, axis=0).astype('float32')  # Convert to float32 as FAISS requires it
    distances, indices = faiss_index.search(embedding, 1)  # Find closest match in FAISS index
    if distances[0][0] < 0.6:  # If the distance is below threshold, it's considered a match
        return names[indices[0][0]]
    else:
        return "Unknown"  # If no match found, return "Unknown"


start_face_recognition()
