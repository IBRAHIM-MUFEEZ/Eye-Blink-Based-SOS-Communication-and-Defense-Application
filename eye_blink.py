import cv2
import dlib
from functions import *
from morse_converter import converter

BLINK_RATIO_THRESHOLD = 5.4

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

        if blink_ratio > BLINK_RATIO_THRESHOLD:
            fl = str(flag)
            cv2.putText(frame, fl, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
            flag += 1
        else:
            if flag > 2 and flag <= 4:
                s += '.'
            if flag > 5 and flag <= 9 :
                s += '-'
            if flag >10 and flag <= 100:
                s += ' '
            flag = 0

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
