import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time
from collections import deque

# ---------------- CONFIG ----------------
running = True
blink_counter = 0
last_blink_time = time.time()
eye_movement_history = deque(maxlen=7)   # smoother movement

# Blink parameters
BLINK_THRESHOLD = 0.2
CONSEC_BLINKS_REQUIRED = 2
TIME_WINDOW = 1  # seconds

# Mediapipe FaceMesh with iris refinement
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True,
                                  max_num_faces=1,
                                  min_detection_confidence=0.7,
                                  min_tracking_confidence=0.7)

screen_width, screen_height = pyautogui.size()

# ---------------- UTILITIES ----------------
def euclidean_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def compute_ear(landmarks, frame_w, frame_h):
    """Eye Aspect Ratio (EAR) for blink detection (using right eye)."""
    p1 = (int(landmarks[362].x * frame_w), int(landmarks[362].y * frame_h))  
    p2 = (int(landmarks[386].x * frame_w), int(landmarks[386].y * frame_h))  
    p3 = (int(landmarks[374].x * frame_w), int(landmarks[374].y * frame_h))  
    p4 = (int(landmarks[263].x * frame_w), int(landmarks[263].y * frame_h))  

    ear = euclidean_distance(p2, p3) / (euclidean_distance(p1, p4) + 1e-6)
    return ear

def get_iris_center(landmarks, frame_w, frame_h):
    """Get average iris center from both eyes."""
    # Right eye iris landmarks: 474–477
    right_eye = [(landmarks[i].x * frame_w, landmarks[i].y * frame_h) for i in [474,475,476,477]]
    # Left eye iris landmarks: 469–472
    left_eye = [(landmarks[i].x * frame_w, landmarks[i].y * frame_h) for i in [469,470,471,472]]

    # Compute centers
    right_center = np.mean(right_eye, axis=0)
    left_center = np.mean(left_eye, axis=0)

    # Final center = midpoint of both pupils
    iris_center = ((right_center[0] + left_center[0]) / 2,
                   (right_center[1] + left_center[1]) / 2)
    return iris_center

# ---------------- MAIN LOOP ----------------
def detect_eye_tracking():
    global running, blink_counter, last_blink_time

    cap = cv2.VideoCapture(0)

    while cap.isOpened() and running:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        frame_h, frame_w, _ = frame.shape

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark

            # -------- Cursor Movement --------
            iris_center = get_iris_center(landmarks, frame_w, frame_h)
            eye_x, eye_y = iris_center[0] / frame_w, iris_center[1] / frame_h

            # Save movement history for smoothing
            eye_movement_history.append((eye_x, eye_y))
            avg_x = np.mean([pt[0] for pt in eye_movement_history])
            avg_y = np.mean([pt[1] for pt in eye_movement_history])

            # Map to screen size
            screen_x = int(avg_x * screen_width)
            screen_y = int(avg_y * screen_height)
            pyautogui.moveTo(screen_x, screen_y, duration=0.05)  # smoother

            # Draw eye center for debugging
            cv2.circle(frame, (int(iris_center[0]), int(iris_center[1])), 5, (0,255,0), -1)

            # -------- Blink Detection --------
            ear = compute_ear(landmarks, frame_w, frame_h)

            if ear < BLINK_THRESHOLD:  
                if time.time() - last_blink_time < TIME_WINDOW:  
                    blink_counter += 1
                else:
                    blink_counter = 1  
                last_blink_time = time.time()

                if blink_counter >= CONSEC_BLINKS_REQUIRED:  
                    pyautogui.click()
                    print("✅ Double Blink → Click!")
                    blink_counter = 0  

        # Show feed
        cv2.imshow("Eye Tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# ---------------- RUN ----------------
if __name__ == "__main__":
    try:
        detect_eye_tracking()
    except KeyboardInterrupt:
        print("Program stopped by user")
        running = False
