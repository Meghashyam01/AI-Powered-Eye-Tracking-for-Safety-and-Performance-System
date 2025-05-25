import cv2
import dlib
import numpy as np

# Load Dlib's face detector and facial landmarks predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

LEFT_EYE = [36, 37, 38, 39, 40, 41]
RIGHT_EYE = [42, 43, 44, 45, 46, 47]

def get_eye_region(landmarks, eye_points):
    """ Extracts the eye region given the facial landmarks """
    return np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in eye_points], np.int32)

def detect_iris_position(eye_region, gray):
    """ Detects the iris inside the eye and determines its position """
    mask = np.zeros_like(gray)
    cv2.fillPoly(mask, [eye_region], 255)
    eye_only = cv2.bitwise_and(gray, gray, mask=mask)

    # Apply thresholding to isolate the iris from the sclera (white part)
    _, thresh = cv2.threshold(eye_only, 30, 255, cv2.THRESH_BINARY_INV)

    # Detect contours of the iris
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        max_contour = max(contours, key=cv2.contourArea)  # Get the largest contour (iris)
        M = cv2.moments(max_contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])  # X-coordinate of iris center
            cy = int(M["m01"] / M["m00"])  # Y-coordinate of iris center
            return (cx, cy), max_contour
    return None, None

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    for face in faces:
        landmarks = predictor(gray, face)

        # Get left and right eye regions
        left_eye = get_eye_region(landmarks, LEFT_EYE)
        right_eye = get_eye_region(landmarks, RIGHT_EYE)

        # Detect iris position
        left_iris, left_contour = detect_iris_position(left_eye, gray)
        right_iris, right_contour = detect_iris_position(right_eye, gray)

        # Draw eye contours
        cv2.polylines(frame, [left_eye], True, (0, 255, 0), 1)
        cv2.polylines(frame, [right_eye], True, (0, 255, 0), 1)

        # Track iris movement and show an alert for LEFT
        if left_iris:
            cx, cy = left_iris
            cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)  # Red dot for iris center

            # Find eye boundaries to compare position
            min_x = np.min(left_eye[:, 0])  # Leftmost point of the eye
            max_x = np.max(left_eye[:, 0])  # Rightmost point of the eye

            if cx < min_x + (max_x - min_x) // 3:  # If iris is in left third of the eye
                cv2.putText(frame, "ALERT: LOOKING LEFT!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                            1, (0, 0, 255), 2, cv2.LINE_AA)

            elif cx > min_x + 2 * (max_x - min_x) // 3:  # If iris is in right third
                cv2.putText(frame, "ALERT: LOOKING RIGHT!", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 
                            1, (255, 0, 0), 2, cv2.LINE_AA)

        # Track iris movement and show an alert for RIGHT eye
        if right_iris:
            cx, cy = right_iris
            cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)  # Red dot for iris center

            min_x = np.min(right_eye[:, 0])  # Leftmost point of the eye
            max_x = np.max(right_eye[:, 0])  # Rightmost point of the eye

            if cx < min_x + (max_x - min_x) // 3:  # If iris is in left third
                cv2.putText(frame, "ALERT: LOOKING LEFT!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                            1, (0, 0, 255), 2, cv2.LINE_AA)

            elif cx > min_x + 2 * (max_x - min_x) // 3:  # If iris is in right third
                cv2.putText(frame, "ALERT: LOOKING RIGHT!", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 
                            1, (255, 0, 0), 2, cv2.LINE_AA)

    cv2.imshow("Eyeball Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

