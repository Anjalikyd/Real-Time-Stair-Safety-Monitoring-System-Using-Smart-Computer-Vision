import cv2
import numpy as np
import pyttsx3
import time
from ultralytics import YOLO
import mediapipe as mp
import threading
import queue

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Initialize text-to-speech engine in a separate thread
tts_queue = queue.Queue()

def tts_worker():
    """
    Worker function for the text-to-speech thread.
    Initializes the engine and processes messages from the queue.
    """
    engine = pyttsx3.init()
    engine.startLoop(False)  # Start a non-blocking loop
    while True:
        try:
            msg = tts_queue.get(timeout=1)
            if msg is None:  # Sentinel value to exit the thread
                engine.endLoop()
                break
            engine.say(msg)
            # We no longer use runAndWait() here.
            # We let engine.iterate() in the main loop handle the speech.
            tts_queue.task_done()
        except queue.Empty:
            # This is important! We call iterate() to process events
            # even when the queue is empty, which prevents the RuntimeError.
            engine.iterate()

tts_thread = threading.Thread(target=tts_worker, daemon=True)
tts_thread.start()

def speak_alert(msg):
    """
    Function to put a message on the queue for the TTS thread to process.
    """
    # Check if the queue is not full to avoid blocking
    if not tts_queue.full():
        tts_queue.put(msg)

# Mediapipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_draw = mp.solutions.drawing_utils

# Define staircase area polygon (manually adjust to your camera view)
staircase_polygon = np.array([
    [100, 200],
    [500, 200],
    [500, 600],
    [100, 600]
])

# Define railing area box (x1, y1, x2, y2)
railing_box = (80, 200, 130, 600)

def point_inside_polygon(x, y, poly):
    return cv2.pointPolygonTest(poly, (x, y), False) >= 0

def point_inside_box(x, y, box):
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2

# Time-based alert control
last_mobile_alert_time = 0
last_railing_alert_time = 0
ALERT_INTERVAL = 5  # seconds

# Open webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    current_mobile_alert = False
    current_railing_alert = False
    current_time = time.time()

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pose_results = pose.process(frame_rgb)
    h, w, _ = frame.shape

    # Object detection
    results = model(frame)[0]
    persons = []
    phones = []

    for box in results.boxes:
        cls_id = int(box.cls[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if cls_id == 0:
            persons.append(((x1, y1, x2, y2), (cx, cy)))
            label = "Person"
            color = (0, 255, 0)
        elif cls_id == 67:
            phones.append(((x1, y1, x2, y2), (cx, cy)))
            label = "Phone"
            color = (0, 0, 255)
        else:
            continue

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4)
        cv2.putText(frame, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Phone usage detection logic
    for person_box, person_center in persons:
        if point_inside_polygon(*person_center, staircase_polygon):
            for phone_box, phone_center in phones:
                px1, py1, px2, py2 = person_box
                fx1, fy1, fx2, fy2 = phone_box

                overlap_x = min(px2, fx2) - max(px1, fx1)
                overlap_y = min(py2, fy2) - max(py1, fy1)
                if overlap_x > 0 and overlap_y > 0:
                    current_mobile_alert = True
                    cv2.putText(frame, "ALERT: Phone Detected!",
                                (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                                1, (0, 0, 255), 3)

    # Railing holding detection using pose
    if pose_results.pose_landmarks:
        mp_draw.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        landmarks = pose_results.pose_landmarks.landmark

        left_wrist = landmarks[15]
        right_wrist = landmarks[16]
        lw_x, lw_y = int(left_wrist.x * w), int(left_wrist.y * h)
        rw_x, rw_y = int(right_wrist.x * w), int(right_wrist.y * h)

        # Draw wrist points
        cv2.circle(frame, (lw_x, lw_y), 5, (255, 0, 255), -1)
        cv2.circle(frame, (rw_x, rw_y), 5, (255, 0, 255), -1)
        
        # Trigger alert if both wrists are outside the railing box
        if not point_inside_box(lw_x, lw_y, railing_box) and not point_inside_box(rw_x, rw_y, railing_box):
            current_railing_alert = True
            cv2.putText(frame, "ALERT: Not Holding Railing!",
                        (50, 100), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 140, 255), 3)

    # Voice alerts (repeat every ALERT_INTERVAL while condition holds)
    if current_mobile_alert and current_railing_alert:
        if current_time - last_mobile_alert_time > ALERT_INTERVAL and current_time - last_railing_alert_time > ALERT_INTERVAL:
            speak_alert("Alert! Do not use the phone and please hold the railing.")
            last_mobile_alert_time = current_time
            last_railing_alert_time = current_time
    elif current_mobile_alert:
        if current_time - last_mobile_alert_time > ALERT_INTERVAL:
            speak_alert("Alert! Please do not use the phone.")
            last_mobile_alert_time = current_time
    elif current_railing_alert:
        if current_time - last_railing_alert_time > ALERT_INTERVAL:
            speak_alert("Alert! Please hold the railing for your safety.")
            last_railing_alert_time = current_time

    # Draw staircase and railing zone
    cv2.polylines(frame, [staircase_polygon], isClosed=True, color=(255, 0, 0), thickness=2)
    cv2.rectangle(frame, (railing_box[0], railing_box[1]), (railing_box[2], railing_box[3]),
                  (0, 255, 255), 2)
    cv2.putText(frame, "Railing", (railing_box[0], railing_box[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.imshow("Staircase Monitor", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
tts_queue.put(None)
tts_thread.join()
cap.release()
cv2.destroyAllWindows()