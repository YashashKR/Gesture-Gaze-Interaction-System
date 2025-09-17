# drawing_canvas.py
import cv2
import mediapipe as mp
import numpy as np
import threading
import time


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


colors = [
    (0, 0, 0), (0, 0, 255), (255, 0, 0), (0, 255, 0),
    (0, 255, 255), (0, 165, 255), (128, 0, 128), (255, 255, 0),
    (255, 192, 203), (128, 128, 0)
]
color_names = ["BLACK","RED","BLUE","GREEN","YELLOW","ORANGE","PURPLE","LT BLUE","PINK","OLIVE"]

current_color = colors[0]
stroke_sizes = [5, 10, 20, 30]
current_stroke = stroke_sizes[1]


sidebar_height = 60   
canvas_height = 720
canvas_width = 1280


palette_size = 60
palette_start_y = 40   

canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255
prev_center = None
running = False
_thread = None



def draw_button_bar_on_frame(frame):
    """Draw top bar (Clear, Eraser)."""
    
    cv2.rectangle(frame, (100, 1), (220, sidebar_height - 1), (0, 0, 0), 2)
    cv2.putText(frame, "CLEAR", (120, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    
    cv2.rectangle(frame, (240, 1), (360, sidebar_height - 1), (0, 0, 0), 2)
    cv2.putText(frame, "ERASER", (255, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    if current_color == (255,255,255): 
        cv2.line(frame, (240, sidebar_height-5), (360, sidebar_height-5), (255,255,255), 3)

def draw_color_palette(frame):
    """Draw vertical palette of available colors on the left side with highlight."""
    for i, clr in enumerate(colors):
        y1 = palette_start_y + i * (palette_size + 10)
        y2 = y1 + palette_size
        x1, x2 = 10, 10 + palette_size

        cv2.rectangle(frame, (x1, y1), (x2, y2), clr, -1)

        
        if clr == current_color:
            cv2.rectangle(frame, (x1-3, y1-3), (x2+3, y2+3), (255,255,255), 5)
        else:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)

def finger_up(lm, tip_idx, pip_idx):
    return lm[tip_idx].y < lm[pip_idx].y

def draw_color_preview(frame):
    """Show the currently selected color as a preview box in the top bar."""
    preview_x1, preview_y1 = canvas_width - 180, 10
    preview_x2, preview_y2 = canvas_width - 100, sidebar_height - 10

    
    cv2.rectangle(frame, (preview_x1, preview_y1), (preview_x2, preview_y2), current_color, -1)
    cv2.rectangle(frame, (preview_x1, preview_y1), (preview_x2, preview_y2), (255,255,255), 2)

    
    cv2.putText(frame, "Selected", (preview_x1, preview_y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)


def _detect_loop():
    global running, prev_center, canvas, current_color, current_stroke

    with mp_hands.Hands(max_num_hands=1,
                        min_detection_confidence=0.75,
                        min_tracking_confidence=0.75) as hands:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ drawing_canvas: Could not open webcam")
            running = False
            return

        cap.set(3, canvas_width)
        cap.set(4, canvas_height)

        print("✅ drawing_canvas: started camera loop")
        prev_center = None

        while running:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            frame = cv2.resize(frame, (canvas_width, canvas_height))

            draw_button_bar_on_frame(frame)
            draw_color_palette(frame)
            draw_color_preview(frame)

            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                lm = hand_landmarks.landmark
                nx, ny = lm[8].x, lm[8].y
                ix, iy = int(nx * w), int(ny * h)

                index_up = finger_up(lm, 8, 6)
                middle_up = finger_up(lm, 12, 10)

                # Top bar
                if iy <= sidebar_height:
                    if 100 <= ix <= 220: canvas[:] = 255
                    elif 240 <= ix <= 360: current_color = (255, 255, 255)

                # Left palette
                elif 10 <= ix <= 10 + palette_size:
                    for i in range(len(colors)):
                        y1 = palette_start_y + i * (palette_size + 10)
                        y2 = y1 + palette_size
                        if y1 <= iy <= y2:
                            current_color = colors[i]
                            print(f"🎨 Color selected: {color_names[i]}")
                            time.sleep(0.25)
                            break

                # Drawing
                else:
                    cx = int(nx * canvas_width)
                    cy = int(ny * canvas_height)
                    if index_up and not middle_up:
                        if prev_center is None:
                            prev_center = (cx, cy)
                        cv2.line(canvas, prev_center, (cx, cy), current_color, current_stroke)
                        prev_center = (cx, cy)
                    else:
                        prev_center = None
            else:
                prev_center = None

            # Masked view
            masked = cv2.addWeighted(frame, 0.3, canvas, 0.7, 0)
            cv2.imshow("Drawing Canvas", masked)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or not running:
                break
            elif key == ord("c"): canvas[:] = 255
            elif key == ord("e"): current_color = (255, 255, 255)
            elif key == ord("1"): current_stroke = stroke_sizes[0]
            elif key == ord("2"): current_stroke = stroke_sizes[1]
            elif key == ord("3"): current_stroke = stroke_sizes[2]
            elif key == ord("4"): current_stroke = stroke_sizes[3]

        cap.release()
        cv2.destroyAllWindows()
        running = False


# ---------------- START/STOP ----------------
def start_canvas():
    global running, _thread
    if running: return
    running = True
    _thread = threading.Thread(target=_detect_loop, daemon=True)
    _thread.start()

def stop_canvas():
    global running
    running = False
