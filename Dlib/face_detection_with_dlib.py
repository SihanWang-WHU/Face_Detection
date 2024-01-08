import cv2
import face_recognition

def detect_faces_and_save_bboxes(video_path, output_txt_path):
    """
    Detects faces in a video file, saves the bounding boxes of each face,
    and writes the bounding box information to a text file.

    Args:
    video_path (str): Path to the video file.
    output_txt_path (str): Path to the output text file.
    """
    # Open the video file
    video_capture = cv2.VideoCapture(video_path)
    if not video_capture.isOpened():
        print("Error: Could not open video file.")
        return

    frame_counter = 0
    bbox_info = []

    # Read frames from the video
    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            break

        # Detect faces in the frame
        faces = face_recognition.face_locations(frame, model="hog")
        bbox_info.append((frame_counter, faces))

        frame_counter += 1

    # Release the video capture object
    video_capture.release()

    # Write bbox info to a text file
    with open(output_txt_path, "w") as file:
        for frame_number, bboxes in bbox_info:
            file.write(f"Frame {frame_number}: {bboxes}\n")

def main():
    # Define the path to the video file and output text file
    video_path = '../test.mp4'
    output_txt_path = 'bbox_output.txt'

    # Detect faces and save bbox information to a text file
    detect_faces_and_save_bboxes(video_path, output_txt_path)
    print(f"Bounding box information saved to {output_txt_path}")

if __name__ == "__main__":
    main()
