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
scroll_cooldown = 0.2  


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

# Alternative scroll function using hand movement
def scroll_with_hand_movement(current_y, previous_y):
    global last_scroll_time
    
    current_time = time.time()
    
    if current_time - last_scroll_time < scroll_cooldown:
        return
    
    if previous_y is not None:
        movement = current_y - previous_y
        
        if abs(movement) > 0.02:  
            if movement > 0:  
                pyautogui.scroll(-5) 
                print("Scrolling down (hand movement)")
            else: 
                pyautogui.scroll(5)   
                print("Scrolling up (hand movement)")
            
            last_scroll_time = current_time

# Hand gesture detection function
def detect_hand_gestures():
    global is_dragging, running
    
    cap = cv2.VideoCapture(0)
    previous_index_y = None  
    
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

                    
                    scroll_screen(index_finger_y, middle_finger_tip.y)
                    
                    
                    if pinch_distance < 0.05:
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
            cv2.putText(image, "Index above/below middle: Scroll", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
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