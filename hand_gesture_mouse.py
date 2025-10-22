import cv2
import mediapipe as mp
import pyautogui
import math
import time
import threading

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

screen_width, screen_height = pyautogui.size()

is_dragging = False
running = True
last_scroll_time = 0
scroll_cooldown = 0.25

# Smoothing parameters
smoothed_points = []
SMOOTHING_WINDOW = 3
SMOOTHING_FACTOR = 0.3  # How much to ease mouse movement (0.1 - slow, 0.5 - fast)
last_screen_x, last_screen_y = pyautogui.position()


def calculate_distance(landmark1, landmark2):
    return math.sqrt((landmark1.x - landmark2.x) ** 2 + (landmark1.y - landmark2.y) ** 2)


def move_mouse(index_finger_x, index_finger_y):
    """Smoothly move the mouse pointer using exponential interpolation."""
    global last_screen_x, last_screen_y

    target_x = int(index_finger_x * screen_width)
    target_y = int(index_finger_y * screen_height)

    # Smooth easing between current and target
    smooth_x = last_screen_x + (target_x - last_screen_x) * SMOOTHING_FACTOR
    smooth_y = last_screen_y + (target_y - last_screen_y) * SMOOTHING_FACTOR

    # Only move if difference is significant (prevents micro jitter)
    if abs(target_x - last_screen_x) > 2 or abs(target_y - last_screen_y) > 2:
        pyautogui.moveTo(smooth_x, smooth_y)

    last_screen_x, last_screen_y = smooth_x, smooth_y


def scroll_screen(index_y, middle_y):
    """Scroll the screen based on vertical finger distance."""
    global last_scroll_time
    current_time = time.time()
    if current_time - last_scroll_time < scroll_cooldown:
        return

    finger_distance = index_y - middle_y

    if abs(finger_distance) > 0.02:
        if finger_distance > 0.01:
            pyautogui.scroll(-30)  # Scroll down
        elif finger_distance < -0.01:
            pyautogui.scroll(30)   # Scroll up

        last_scroll_time = current_time


def detect_hand_gestures():
    """Main hand detection loop."""
    global is_dragging, running, smoothed_points

    print("Initializing Camera...")
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)  # Reduce resolution for smoother FPS
    cap.set(4, 480)
    print("Camera Initialized")
    previous_index_y = None

    with mp_hands.Hands(max_num_hands=1,
                        min_detection_confidence=0.8,
                        min_tracking_confidence=0.8) as hands:

        while cap.isOpened() and running:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False
            results = hands.process(rgb_frame)
            rgb_frame.flags.writeable = True
            image = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                    middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

                    # Collect smoothed coordinates
                    current_point = (index_finger_tip.x, index_finger_tip.y)
                    smoothed_points.append(current_point)
                    if len(smoothed_points) > SMOOTHING_WINDOW:
                        smoothed_points.pop(0)

                    avg_x = sum(p[0] for p in smoothed_points) / len(smoothed_points)
                    avg_y = sum(p[1] for p in smoothed_points) / len(smoothed_points)

                    move_mouse(avg_x, avg_y)
                    scroll_screen(index_finger_tip.y, middle_finger_tip.y)

                    # Detect pinch for click/drag
                    pinch_distance = calculate_distance(index_finger_tip, thumb_tip)
                    if pinch_distance < 0.05:
                        if not is_dragging:
                            pyautogui.mouseDown()
                            is_dragging = True
                    else:
                        if is_dragging:
                            pyautogui.mouseUp()
                            is_dragging = False

            else:
                smoothed_points.clear()

            # UI overlays
            cv2.putText(image, "q: Quit", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(image, "Pinch: Click/Drag", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(image, "Index above/below middle: Scroll", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.imshow("Hand Gesture Detection", image)

            # Slight delay for CPU balance
            time.sleep(0.01)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                running = False
                break

    cap.release()
    cv2.destroyAllWindows()


# Flask integration functions
def start_hand_gesture():
    global running
    if running:
        return
    running = True
    thread = threading.Thread(target=detect_hand_gestures, daemon=True)
    thread.start()


def stop_hand_gesture():
    global running
    running = False


if __name__ == "__main__":
    try:
        detect_hand_gestures()
    except KeyboardInterrupt:
        print("Program stopped by user")
        running = False
