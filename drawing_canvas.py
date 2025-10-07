# drawing_canvas.py
import cv2
import mediapipe as mp
import numpy as np
import threading
import time
import datetime

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# ---------------- CONFIG ----------------
colors = [
    (0, 0, 0), (0, 0, 255), (255, 0, 0), (0, 255, 0),
    (0, 255, 255), (0, 165, 255), (128, 0, 128), (255, 255, 0),
    (255, 192, 203), (128, 128, 0)
]
color_names = ["BLACK","RED","BLUE","GREEN","YELLOW","ORANGE","PURPLE","LT BLUE","PINK","OLIVE"]

current_color = colors[0]
stroke_sizes = [5, 10, 20, 30]
current_stroke = stroke_sizes[1]

# Stroke styles
stroke_styles = ["normal", "dashed", "zigzag", "straight"]
current_style_idx = 0

# UI / canvas sizes
sidebar_height = 60
canvas_height = 720
canvas_width = 1280
palette_size = 50
palette_start_y = 40

# Canvas and state
canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255
prev_center = None
running = False
_thread = None
dark_mode = False
show_info = False

# History for undo: snapshots of canvas BEFORE a stroke begins
history = []

# drawing state & debounce
is_drawing = False
last_action_time = 0.0
ACTION_COOLDOWN = 0.45  # seconds

# Stabilization variables
smoothed_points = []
SMOOTHING_WINDOW = 5 # Number of points to average for stabilization

# ---------------- helpers ----------------
def finger_up(lm, tip_idx, pip_idx):
    return lm[tip_idx].y < lm[pip_idx].y

def draw_button_bar_on_frame(frame, fingertip=None):
    """Top buttons: smaller, colored, with hover glow when fingertip is over them."""
    buttons = [
        ("CLEAR", (100, 10, 180, 50), (200, 50, 50), "clear"),
        ("ERASER", (200, 10, 280, 50), (70, 70, 70), "eraser"),
        ("UNDO", (300, 10, 380, 50), (50, 120, 200), "undo"),
        ("INFO", (400, 10, 480, 50), (50, 200, 100), "info"),
        ("SAVE", (500, 10, 580, 50), (255, 255, 0), "save")
    ]
    for text, (x1, y1, x2, y2), color, mode in buttons:
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1, lineType=cv2.LINE_AA)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 1, lineType=cv2.LINE_AA)

        if fingertip and x1 <= fingertip[0] <= x2 and y1 <= fingertip[1] <= y2:
            cv2.rectangle(frame, (x1-4, y1-4), (x2+4, y2+4), (255, 255, 255), 3, lineType=cv2.LINE_AA)
            cv2.rectangle(frame, (x1-8, y1-8), (x2+8, y2+8), (255, 255, 255), 1, lineType=cv2.LINE_AA)

        cv2.putText(frame, text, (x1+8, y1+30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, lineType=cv2.LINE_AA)

def draw_color_palette(frame):
    """Vertical palette on the left (larger boxes)."""
    for i, clr in enumerate(colors):
        y1 = palette_start_y + i * (palette_size + 10)
        y2 = y1 + palette_size
        x1, x2 = 10, 10 + palette_size
        cv2.rectangle(frame, (x1, y1), (x2, y2), clr, -1, lineType=cv2.LINE_AA)
        if clr == current_color:
            cv2.rectangle(frame, (x1-4, y1-4), (x2+4, y2+4), (255,255,255), 4, lineType=cv2.LINE_AA)
        else:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2, lineType=cv2.LINE_AA)

def draw_color_preview(frame):
    """Preview of selected color (top-right)."""
    preview_x1, preview_y1 = canvas_width - 180, 10
    preview_x2, preview_y2 = canvas_width - 100, sidebar_height - 10
    cv2.rectangle(frame, (preview_x1, preview_y1), (preview_x2, preview_y2), current_color, -1, lineType=cv2.LINE_AA)
    cv2.rectangle(frame, (preview_x1, preview_y1), (preview_x2, preview_y2), (255,255,255), 1, lineType=cv2.LINE_AA)
    cv2.putText(frame, stroke_styles[current_style_idx], (preview_x1 - 10, preview_y2 + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, lineType=cv2.LINE_AA)

def draw_info_card(frame):
    """Larger info card so every line fits (only if toggled)."""
    if not show_info:
        return
    card_w = 360
    card_h = 260
    x1, y1 = canvas_width - card_w - 20, 80
    x2, y2 = x1 + card_w, y1 + card_h
    overlay_color = (30, 30, 30) if dark_mode else (240, 240, 240)
    text_color = (255, 255, 255) if dark_mode else (0, 0, 0)
    cv2.rectangle(frame, (x1, y1), (x2, y2), overlay_color, -1, lineType=cv2.LINE_AA)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (180, 180, 180), 1, lineType=cv2.LINE_AA)

    lines = [
        "CONTROLS",
        "1-4 : Change stroke size",
        "S : Cycle stroke style (normal/dashed/zigzag/straight)",
        "E : Eraser (sets color to white/black depending on mode)",
        "C : Clear canvas",
        "U or UNDO button : Undo last stroke",
        "M : Toggle Dark / Light (clears canvas to mode background)",
        "I or INFO button : Toggle this info card",
        "A or SAVE button : Save canvas",
        "Q : Quit canvas"
    ]
    for i, text in enumerate(lines):
        cv2.putText(frame, text, (x1+12, y1+30 + i*25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, text_color, 1, lineType=cv2.LINE_AA)

def draw_styled_line(img, p1, p2, color, thickness, style):
    """Draw line with different styles between two points."""
    if style == "normal":
        cv2.line(img, p1, p2, color, thickness, lineType=cv2.LINE_AA)
    elif style == "dashed":
        dist = int(np.hypot(p2[0]-p1[0], p2[1]-p1[1]))
        if dist == 0:
            return
        # draw short segments/dashes
        dash_len = 18
        gap = 8
        for start in range(0, dist, dash_len + gap):
            t0 = start / dist
            t1 = min((start + dash_len) / dist, 1.0)
            sx = int(p1[0] + (p2[0] - p1[0]) * t0)
            sy = int(p1[1] + (p2[1] - p1[1]) * t0)
            ex = int(p1[0] + (p2[0] - p1[0]) * t1)
            ey = int(p1[1] + (p2[1] - p1[1]) * t1)
            cv2.line(img, (sx, sy), (ex, ey), color, thickness, lineType=cv2.LINE_AA)
    elif style == "zigzag":
        # simple two-segment zigzag
        mx = (p1[0] + p2[0]) // 2
        my = (p1[1] + p2[1]) // 2
        offset = max(8, thickness * 2)
        cv2.line(img, p1, (mx, my - offset), color, thickness, lineType=cv2.LINE_AA)
        cv2.line(img, (mx, my - offset), p2, color, thickness, lineType=cv2.LINE_AA)
    elif style == "straight":
        # straight with 1px for a clean line
        cv2.line(img, p1, p2, color, 1, lineType=cv2.LINE_AA)

# ---------------- MAIN LOOP ----------------
def _detect_loop():
    global running, prev_center, canvas, current_color, current_stroke
    global dark_mode, show_info, is_drawing, last_action_time, current_style_idx, smoothed_points

    # Ensure canvas initial background matches current mode
    canvas[:] = 0 if dark_mode else 255

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

        prev_center = None
        is_drawing = False

        while running:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            frame = cv2.resize(frame, (canvas_width, canvas_height))

            draw_color_palette(frame)
            draw_color_preview(frame)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            fingertip = None
            ix = iy = None

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                lm = hand_landmarks.landmark
                w, h = frame.shape[1], frame.shape[0]
                nx, ny = lm[8].x, lm[8].y
                ix, iy = int(nx * w), int(ny * h)
                fingertip = (ix, iy)

                index_up = finger_up(lm, 8, 6)
                middle_up = finger_up(lm, 12, 10)

                # Stabilization: add new point to the list
                smoothed_points.append(fingertip)
                if len(smoothed_points) > SMOOTHING_WINDOW:
                    smoothed_points.pop(0)
                
                # Calculate the stabilized point
                stabilized_x = int(sum(p[0] for p in smoothed_points) / len(smoothed_points))
                stabilized_y = int(sum(p[1] for p in smoothed_points) / len(smoothed_points))
                stabilized_fingertip = (stabilized_x, stabilized_y)

                # TOP BAR interactions (debounced)
                if iy <= sidebar_height:
                    now = time.time()
                    if (now - last_action_time) > ACTION_COOLDOWN:
                        # Check buttons for interaction
                        buttons = [
                            ("CLEAR", (100, 10, 180, 50), "clear"),
                            ("ERASER", (200, 10, 280, 50), "eraser"),
                            ("UNDO", (300, 10, 380, 50), "undo"),
                            ("INFO", (400, 10, 480, 50), "info"),
                            ("SAVE", (500, 10, 580, 50), "save")
                        ]
                        for text, (x1, y1, x2, y2), action in buttons:
                            if x1 <= ix <= x2 and y1 <= iy <= y2:
                                if action == "clear":
                                    canvas[:] = 0 if dark_mode else 255
                                    history.clear()
                                elif action == "eraser":
                                    current_color = (0,0,0) if dark_mode else (255,255,255)
                                elif action == "undo":
                                    if history:
                                        canvas[:] = history.pop()
                                elif action == "info":
                                    show_info = not show_info
                                elif action == "save":
                                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                                    filename = f"canvas_{timestamp}.png"
                                    cv2.imwrite(filename, canvas)
                                    print(f"✅ Saved canvas as {filename}")
                                last_action_time = now
                                break

                # LEFT palette selection (debounced)
                elif 10 <= ix <= 10 + palette_size:
                    now = time.time()
                    for i in range(len(colors)):
                        y1 = palette_start_y + i * (palette_size + 10)
                        y2 = y1 + palette_size
                        if y1 <= iy <= y2 and (now - last_action_time) > ACTION_COOLDOWN:
                            current_color = colors[i]
                            last_action_time = now
                            break

                # DRAWING AREA
                else:
                    cx, cy = stabilized_fingertip

                    if index_up and not middle_up:
                        if not is_drawing:
                            history.append(canvas.copy())
                            is_drawing = True
                            prev_center = (cx, cy)
                        else:
                            draw_styled_line(canvas, prev_center, (cx, cy), current_color, current_stroke, stroke_styles[current_style_idx])
                            prev_center = (cx, cy)
                    else:
                        is_drawing = False
                        prev_center = None
                        smoothed_points.clear()

            else:
                is_drawing = False
                prev_center = None
                smoothed_points.clear()

            draw_button_bar_on_frame(frame, fingertip)

            if show_info:
                draw_info_card(frame)

            if dark_mode:
                masked = cv2.addWeighted(frame, 0.15, canvas, 0.85, 0)
            else:
                masked = cv2.addWeighted(frame, 0.30, canvas, 0.70, 0)

            cv2.imshow("Drawing Canvas", masked)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or not running:
                break
            elif key == ord("c"):
                canvas[:] = 0 if dark_mode else 255
                history.clear()
            elif key == ord("e"):
                current_color = (0,0,0) if dark_mode else (255,255,255)
            elif key == ord("1"):
                current_stroke = stroke_sizes[0]
            elif key == ord("2"):
                current_stroke = stroke_sizes[1]
            elif key == ord("3"):
                current_stroke = stroke_sizes[2]
            elif key == ord("4"):
                current_stroke = stroke_sizes[3]
            elif key == ord("m"):
                dark_mode = not dark_mode
                canvas[:] = 0 if dark_mode else 255
                history.clear()
            elif key == ord("s"):
                current_style_idx = (current_style_idx + 1) % len(stroke_styles)
            elif key == ord("u"):
                if history:
                    canvas[:] = history.pop()
            elif key == ord("i"):
                show_info = not show_info
            elif key == ord("a"): # New keyboard shortcut for save
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"canvas_{timestamp}.png"
                cv2.imwrite(filename, canvas)
                print(f"✅ Saved canvas as {filename}")


        cap.release()
        cv2.destroyAllWindows()
        running = False

# ---------------- START/STOP ----------------
def start_canvas():
    global running, _thread
    if running:
        return
    running = True
    _thread = threading.Thread(target=_detect_loop, daemon=True)
    _thread.start()

def stop_canvas():
    global running
    running = False