import cv2
import dlib
from functions import *
from morse_converter import converter

# Define the blink thresholds for dot, dash, and space
DOT_BLINK_THRESHOLD = 5.4
DASH_BLINK_THRESHOLD = 10.8
SPACE_BLINK_THRESHOLD = 16.2

# Initialize the webcam capture using OpenCV
cap = cv2.VideoCapture(0)

# Create a display window using OpenCV
cv2.namedWindow('DECODE')

# Load the face detection model from dlib
detector = dlib.get_frontal_face_detector()

# Load the facial landmark predictor model from dlib
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Define the indices of the left and right eye landmarks
left_eye_landmarks = [36, 37, 38, 39, 40, 41]
right_eye_landmarks = [42, 43, 44, 45, 46, 47]

flag = 0
s = ''

while True:
    # Capture a frame from the webcam
    retval, frame = cap.read()

    # Exit the application if no frame is received
    if not retval:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Convert the frame to grayscale
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame using dlib
    faces = detector(frame_gray)

    for face in faces:
        landmarks = predictor(frame_gray, face)

        left_eye_ratio = get_blink_ratio(left_eye_landmarks, landmarks)
        right_eye_ratio = get_blink_ratio(right_eye_landmarks, landmarks)
        blink_ratio = (left_eye_ratio + right_eye_ratio) / 2

        if blink_ratio > SPACE_BLINK_THRESHOLD:
            s += ' '  # 3 blinks for a space
        elif blink_ratio > DASH_BLINK_THRESHOLD:
            s += '-'  # 2 blinks for a dash
        else:
            s += '.'  # 1 blink for a dot

        cv2.putText(frame, s, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "Input Morse: {}".format(s), (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow('DECODE', frame)
    key = cv2.waitKey(1)

    if key == 27:  # Press 'Esc' to exit
        break
    elif key == 32:  # Press 'Space' to clear the Morse code string
        s = ''
    elif key == 13:  # Press 'Enter' to convert Morse code
        decoded_text = converter(s)
        print("Decoded Text: {}".format(decoded_text))
        s = ''

# Release the VideoCapture object
cap.release()
cv2.destroyAllWindows()
