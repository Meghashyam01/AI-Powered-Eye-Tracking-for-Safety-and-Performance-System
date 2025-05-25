import cv2
import dlib
from imutils import face_utils
from scipy.spatial import distance as dist
from playsound import playsound
import threading

# EAR thresholds and frame counters
EYE_AR_THRESH = 0.25
BLINK_FRAMES = 3
DROWSY_FRAMES = 45

# State trackers
blink_counter = 0
drowsy_counter = 0
blinking = False
alarm_on = False
blink_total = 0  # Count total blinks

# Alarm sound
def sound_alarm():
    playsound("Alarm-chosic.com_.mp3")

# EAR calculation
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Load face detector and predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Landmark indices
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)

    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        avgEAR = (leftEAR + rightEAR) / 2.0

        # Draw contours
        cv2.drawContours(frame, [cv2.convexHull(leftEye)], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [cv2.convexHull(rightEye)], -1, (0, 255, 0), 1)

        # EAR below threshold
        if avgEAR < EYE_AR_THRESH:
            blink_counter += 1
            drowsy_counter += 1

            if blink_counter == BLINK_FRAMES:
                blink_total += 1
                blinking = True

            if drowsy_counter >= DROWSY_FRAMES:
                cv2.putText(frame, "DROWSINESS DETECTED!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                if not alarm_on:
                    alarm_on = True
                    threading.Thread(target=sound_alarm).start()

        else:
            blink_counter = 0
            drowsy_counter = 0
            blinking = False
            alarm_on = False

        # Display EAR and blink count
        cv2.putText(frame, f"EAR: {avgEAR:.2f}", (400, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(frame, f"Blinks: {blink_total}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    cv2.imshow("Blink & Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
