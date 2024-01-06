import cv2
import face_recognition

def detect_and_crop_faces(video_path):
    # Load the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file")
        return []

    face_arrays = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Detect faces
        face_locations = face_recognition.face_locations(frame, model="hog")

        for face_location in face_locations:
            top, right, bottom, left = face_location

            # Crop the face
            face_image = frame[top:bottom, left:right]

            # Optionally, resize the face to a standard size, e.g., 224x224 for ResNet
            face_image = cv2.resize(face_image, (224, 224))

            # Convert to array and store
            face_arrays.append(face_image)

    cap.release()
    return face_arrays

# Example usage
video_path = 'test.mp4'
faces = detect_and_crop_faces(video_path)
print(faces)

# 'faces' now contains an array of face images
