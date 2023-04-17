import RPi.GPIO as GPIO
import time
from scipy.spatial import distance
from imutils import face_utils
import imutils
import dlib
import cv2
from pygame import mixer

mixer.init()
mixer.music.load('alarm.wav')

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

thresh = 0.25
frame_check = 10
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]

# Set up GPIO pins for motor control
enable_pin = 18
input1_pin = 23
input2_pin = 24
GPIO.setmode(GPIO.BCM)
GPIO.setup(enable_pin, GPIO.OUT)
GPIO.setup(input1_pin, GPIO.OUT)
GPIO.setup(input2_pin, GPIO.OUT)
motor_pwm = GPIO.PWM(enable_pin, 100)
motor_pwm.start(0)

# Set up GPIO pins for drowsiness detection
drowsy_pin = 17
GPIO.setup(drowsy_pin, GPIO.IN)

# Define a function to control the motor speed
def set_motor_speed(speed):
    if speed > 0:
        GPIO.output(input1_pin, GPIO.HIGH)
        GPIO.output(input2_pin, GPIO.LOW)
    else:
        GPIO.output(input1_pin, GPIO.LOW)
        GPIO.output(input2_pin, GPIO.HIGH)
    motor_pwm.ChangeDutyCycle(abs(speed))

# Define a function to detect drowsiness
def is_drowsy(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    subjects = detect(gray, 0)
    for subject in subjects:
        shape = predict(gray, subject)
        shape = face_utils.shape_to_np(shape)
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0
        if ear < thresh:
            return True
    return False
cap=cv2.VideoCapture(0)
flag=0
# Main loop to control the motor speed based on drowsiness
while True:
    ret, frame = cap.read()
    frame = imutils.resize(frame, width=200)
    if is_drowsy(frame):
        set_motor_speed(10) # set motor speed to 50%
        cv2.putText(frame, "****************ALERT!****************", (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "****************ALERT!****************", (10,325),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        mixer.music.play(-1)
        print("Drowsy detected, slowing down motor")
    else:
        set_motor_speed(100) # set motor speed to 100%
        print("No drowsiness detected, motor at full speed")
    
# Wait for 1 millisecond
    cv2.waitKey(1)

# Check for keyboard interrupt
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
cv2.destroyAllWindows()
cap.release()
set_motor_speed(0)
motor_pwm.stop()
GPIO.cleanup()

