import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time
from collections import deque

running = True
blink_counter = 0
last_blink_time = time.time()
eye_movement_history = deque(maxlen=7)
last_scroll_time = 0

BLINK_THRESHOLD = 0.2
WINK_THRESHOLD = 0.15
CONSEC_BLINKS_REQUIRED = 2
TIME_WINDOW = 1
SCROLL_COOLDOWN = 0.2

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    refine_landmarks=True,
    max_num_faces=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

screen_width, screen_height = pyautogui.size()

def euclidean_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def compute_ear(landmarks, frame_w, frame_h, eye='both'):
    """Eye Aspect Ratio for both eyes separately."""
    if eye == 'left':
        p1 = (landmarks[33].x * frame_w, landmarks[33].y * frame_h)
        p2 = (landmarks[159].x * frame_w, landmarks[159].y * frame_h)
        p3 = (landmarks[145].x * frame_w, landmarks[145].y * frame_h)
        p4 = (landmarks[133].x * frame_w, landmarks[133].y * frame_h)
    elif eye == 'right':
        p1 = (landmarks[362].x * frame_w, landmarks[362].y * frame_h)
        p2 = (landmarks[386].x * frame_w, landmarks[386].y * frame_h)
        p3 = (landmarks[374].x * frame_w, landmarks[374].y * frame_h)
        p4 = (landmarks[263].x * frame_w, landmarks[263].y * frame_h)
    else:  # both eyes average
        left_ear = compute_ear(landmarks, frame_w, frame_h, 'left')
        right_ear = compute_ear(landmarks, frame_w, frame_h, 'right')
        return (left_ear + right_ear) / 2
    
    A = euclidean_distance(p2, p3)
    C = euclidean_distance(p1, p4)
    return A / (C + 1e-6)

def get_iris_center(landmarks, frame_w, frame_h):
    """Get average iris center from both eyes."""
    right_eye = [(landmarks[i].x * frame_w, landmarks[i].y * frame_h) for i in [474,475,476,477]]
    left_eye = [(landmarks[i].x * frame_w, landmarks[i].y * frame_h) for i in [469,470,471,472]]

    right_center = np.mean(right_eye, axis=0)
    left_center = np.mean(left_eye, axis=0)

    iris_center = ((right_center[0] + left_center[0]) / 2,
                   (right_center[1] + left_center[1]) / 2)
    return iris_center

def detect_eye_tracking():
    global running, blink_counter, last_blink_time, last_scroll_time

    cap = cv2.VideoCapture(0)

    while cap.isOpened() and running:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        frame_h, frame_w, _ = frame.shape
        current_time = time.time()

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark

            iris_center = get_iris_center(landmarks, frame_w, frame_h)
            eye_x, eye_y = iris_center[0] / frame_w, iris_center[1] / frame_h

            eye_movement_history.append((eye_x, eye_y))
            avg_x = np.mean([pt[0] for pt in eye_movement_history])
            avg_y = np.mean([pt[1] for pt in eye_movement_history])

            screen_x = int(avg_x * screen_width)
            screen_y = int(avg_y * screen_height)
            pyautogui.moveTo(screen_x, screen_y, duration=0.05)

            cv2.circle(frame, (int(iris_center[0]), int(iris_center[1])), 5, (0,255,0), -1)

            # Get EAR for both eyes
            left_ear = compute_ear(landmarks, frame_w, frame_h, 'left')
            right_ear = compute_ear(landmarks, frame_w, frame_h, 'right')
            avg_ear = (left_ear + right_ear) / 2

            # Left eye wink (close left eye) = scroll down
            if left_ear < WINK_THRESHOLD and right_ear > BLINK_THRESHOLD:
                if current_time - last_scroll_time > SCROLL_COOLDOWN:
                    pyautogui.scroll(-35)
                    print("👁️ Left Eye Wink → Scroll Down")
                    last_scroll_time = current_time

            # Right eye wink (close right eye) = scroll up
            elif right_ear < WINK_THRESHOLD and left_ear > BLINK_THRESHOLD:
                if current_time - last_scroll_time > SCROLL_COOLDOWN:
                    pyautogui.scroll(35)
                    print("👁️ Right Eye Wink → Scroll Up")
                    last_scroll_time = current_time

            # Double blink for click (both eyes)
            elif avg_ear < BLINK_THRESHOLD:
                if current_time - last_blink_time < TIME_WINDOW:
                    blink_counter += 1
                else:
                    blink_counter = 1
                last_blink_time = current_time

                if blink_counter >= CONSEC_BLINKS_REQUIRED:
                    pyautogui.click()
                    print("✅ Double Blink → Click!")
                    blink_counter = 0

        # Show instructions
        cv2.putText(frame, "Left Eye Wink = Scroll Down | Right Eye Wink = Scroll Up", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, "Double Blink = Click | Q = Quit", 
                   (10, frame_h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("Eye Wink Tracker", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        detect_eye_tracking()
    except KeyboardInterrupt:
        print("Program stopped by user")
        running = False