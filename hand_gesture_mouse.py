import cv2
import mediapipe as mp
import pyautogui
import math
import time


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


screen_width, screen_height = pyautogui.size()


is_dragging = False
running = True  
last_scroll_time = 0  
scroll_cooldown = 0.05
baseline_y = None
position_threshold = 0.08
scroll_direction = None
is_continuous_scrolling = False  


def calculate_distance(landmark1, landmark2):
    return math.sqrt((landmark1.x - landmark2.x) ** 2 + (landmark1.y - landmark2.y) ** 2)


def move_mouse(index_finger_x, index_finger_y):
    screen_x = int(index_finger_x * screen_width)
    screen_y = int(index_finger_y * screen_height)
    pyautogui.moveTo(screen_x, screen_y)
    print(f"Mouse moved to: ({screen_x}, {screen_y})")

# Fixed scrolling function
def scroll_screen(index_y, middle_y):
    global last_scroll_time
    
    current_time = time.time()
    
    # Add cooldown to prevent excessive scrolling
    if current_time - last_scroll_time < scroll_cooldown:
        return
    
    # Calculate vertical distance between fingers
    finger_distance = index_y - middle_y
    
    # Debug: Print finger positions and distance
    print(f"Index Y: {index_y:.3f}, Middle Y: {middle_y:.3f}, Distance: {finger_distance:.3f}")
    
    # Adjust sensitivity threshold
    if abs(finger_distance) > 0.02:  
        if finger_distance > 0.01:  
            pyautogui.scroll(-2)  
            print("Scrolling down - Index below middle")
        elif finger_distance < -0.01:  
            pyautogui.scroll(2)  
            print("Scrolling up - Index above middle")
        
        last_scroll_time = current_time

def detect_hand_position(current_y):
    global baseline_y, scroll_direction, is_continuous_scrolling
    
    # Set baseline on first detection
    if baseline_y is None:
        baseline_y = current_y
        return
    
    # Calculate position relative to baseline
    position_diff = current_y - baseline_y
    
    # Check if hand is in up position
    if position_diff < -position_threshold:
        if scroll_direction != "up":
            scroll_direction = "up"
            is_continuous_scrolling = True
            print("Hand UP - Scrolling UP")
    # Check if hand is in down position
    elif position_diff > position_threshold:
        if scroll_direction != "down":
            scroll_direction = "down"
            is_continuous_scrolling = True
            print("Hand DOWN - Scrolling DOWN")
    # Hand is in neutral position
    else:
        if is_continuous_scrolling:
            is_continuous_scrolling = False
            scroll_direction = None
            print("Hand NEUTRAL - Stopped scrolling")

def is_fist_closed(hand_landmarks):
    # Check if all fingertips are below their respective knuckles (closed fist)
    fingertips = [8, 12, 16, 20]  # Index, Middle, Ring, Pinky tips
    knuckles = [6, 10, 14, 18]    # Corresponding knuckles
    
    closed_fingers = 0
    for tip, knuckle in zip(fingertips, knuckles):
        if hand_landmarks.landmark[tip].y > hand_landmarks.landmark[knuckle].y:
            closed_fingers += 1
    
    # Thumb check (different logic)
    thumb_tip = hand_landmarks.landmark[4]
    thumb_knuckle = hand_landmarks.landmark[3]
    if abs(thumb_tip.x - thumb_knuckle.x) < 0.02:  # Thumb close to palm
        closed_fingers += 1
    
    return closed_fingers >= 4  # At least 4 fingers closed = fist

def continuous_scroll(is_fist=False):
    global last_scroll_time
    
    if not is_continuous_scrolling:
        return
        
    current_time = time.time()
    
    if current_time - last_scroll_time < scroll_cooldown:
        return
    
    # Faster scrolling when palm is closed (fist)
    scroll_amount = 8 if is_fist else 4
    
    if scroll_direction == "down":
        pyautogui.scroll(-scroll_amount)
    elif scroll_direction == "up":
        pyautogui.scroll(scroll_amount)
    
    last_scroll_time = current_time

# Hand gesture detection function
def detect_hand_gestures():
    global is_dragging, running, baseline_y
    
    cap = cv2.VideoCapture(0)  
    
    with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.8, min_tracking_confidence=0.8) as hands:
        while cap.isOpened() and running:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            frame = cv2.flip(frame, 1)  
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = hands.process(image)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    # Finger landmarks
                    index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                    middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

                    index_finger_x = index_finger_tip.x
                    index_finger_y = index_finger_tip.y

                    # Distance between index finger and thumb for pinch gesture
                    pinch_distance = calculate_distance(index_finger_tip, thumb_tip)

                    # Detect hand position for scrolling
                    detect_hand_position(index_finger_y)
                    
                    # Check if palm is closed (fist)
                    fist_closed = is_fist_closed(hand_landmarks)
                    
                    # Continue scrolling based on position (faster if fist)
                    continuous_scroll(fist_closed)
                    
                    # Pinch gesture for clicking/dragging (only when not scrolling)
                    if pinch_distance < 0.05 and not is_continuous_scrolling:
                        if not is_dragging:
                            pyautogui.mouseDown()
                            is_dragging = True
                            print("Mouse down (pinch detected)")
                    else:
                        if is_dragging:
                            pyautogui.mouseUp()
                            is_dragging = False
                            print("Mouse up (pinch released)")

                    # Move cursor
                    move_mouse(index_finger_x, index_finger_y)

            # Display instructions on screen
            cv2.putText(image, "q: Quit", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(image, "Pinch: Click/Drag", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(image, "Hold up/down: Scroll, Fist: Fast scroll", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Show scroll status
            if is_continuous_scrolling:
                if results.multi_hand_landmarks:
                    hand_landmarks = results.multi_hand_landmarks[0]
                    fist_closed = is_fist_closed(hand_landmarks)
                    
                    if fist_closed:
                        status_text = f"FAST Scrolling {scroll_direction.upper()}"
                        cv2.putText(image, status_text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    else:
                        status_text = f"Scrolling {scroll_direction.upper()}"
                        cv2.putText(image, status_text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                cv2.putText(image, "Hand NEUTRAL", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            cv2.imshow("Hand Gesture Detection", image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                running = False
                break

    cap.release()
    cv2.destroyAllWindows()

# Run the gesture detection
if __name__ == "__main__":
    try:
        detect_hand_gestures()
    except KeyboardInterrupt:
        print("Program stopped by user")
        running = False