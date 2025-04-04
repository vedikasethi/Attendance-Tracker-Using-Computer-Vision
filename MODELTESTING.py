import os
import cv2
import numpy as np
from keras_facenet import FaceNet
from mtcnn import MTCNN
import csv
from datetime import datetime
import faiss

# Initialize the MTCNN and FaceNet models
detector = MTCNN()
embedder = FaceNet()

# Define paths for embeddings and attendance logs
EMBEDDINGS_PATH = "path_to_load/face_embeddings.csv"  # This is the path to the training data (embeddings)
ATTENDANCE_LOG_PATH = "path_to_save/attendance_log.csv"  # This is the path where attendance logs are saved

# Function to load embeddings from the specified path
def load_embeddings():
    embeddings = []
    names = []
    
    # Load the embeddings from the CSV file specified by EMBEDDINGS_PATH
    with open(EMBEDDINGS_PATH, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            name = row[0]  # The name with SAP ID (from training phase)
            embedding = np.array(row[1:], dtype=float)  # The embedding vector
            embeddings.append(embedding)
            names.append(name)

    embeddings = np.array(embeddings)  # Convert embeddings list to a numpy array
    index = faiss.IndexFlatL2(embeddings.shape[1])  # Create FAISS index using L2 distance metric
    index.add(embeddings)  # Add the embeddings to the FAISS index for fast search
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
            face_images = []  # List to store face images for embedding generation
            for face in faces:
                # Extract the coordinates of the bounding box around the detected face
                x1, y1, width, height = face['box']
                x1, y1 = abs(x1), abs(y1)
                x2, y2 = x1 + width, y1 + height
                face_img = frame[y1:y2, x1:x2]
                face_img = cv2.resize(face_img, (160, 160))  # Resize face to 160x160
                face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)  # Convert to RGB for FaceNet
                face_images.append(face_img)

            # Generate embeddings for the detected faces
            embeddings = embedder.embeddings(face_images)

            # Compare each face embedding with the stored embeddings
            for i, embedding in enumerate(embeddings):
                name = find_closest_match(embedding, faiss_index, names)  # Find closest match using FAISS

                if name != "Unknown" and name not in attendance_marked:
                    # Mark attendance if the person is recognized
                    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    with open(ATTENDANCE_LOG_PATH, mode='a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow([name, timestamp])  # Save name and timestamp to attendance log

                    attendance_marked.add(name)  # Mark attendance for this person
                    print(f"Attendance marked for {name} at {timestamp}.")
                elif name == "Unknown":
                    print("Unknown face detected.")

                # Draw bounding box around the detected face and display the name
                x1, y1, width, height = faces[i]['box']
                cv2.rectangle(frame, (x1, y1), (x2, y1 + height), (0, 255, 0), 2)
                cv2.putText(frame, f"{name}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Display the video frame with detected faces and names
        cv2.imshow("Face Recognition", frame)

        # Exit if 'q' key is pressed
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

# Run face recognition to mark attendance
start_face_recognition()
