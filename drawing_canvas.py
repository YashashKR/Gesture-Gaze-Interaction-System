# drawing_canvas.py - Optimized & Transparent UI Version
import cv2
import mediapipe as mp
import numpy as np
import threading
import time
import datetime
import math
from collections import deque

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# ---------------- CONFIG ----------------
colors = [
    (0, 0, 0), (0, 0, 255), (255, 0, 0), (0, 255, 0),
    (0, 255, 255), (0, 165, 255), (128, 0, 128),
    (255, 255, 0), (255, 192, 203), (255, 255, 255)
]
color_names = ["Black", "Red", "Blue", "Green", "Yellow", "Orange", "Purple", "Cyan", "Pink", "White"]

pen_types = ["pen", "ink", "pencil", "marker", "eraser"]
stroke_styles = ["solid", "dashed", "dotted", "zigzag"]
shapes = ["triangle", "circle", "rectangle", "square"]
size_options = [5, 10, 25, 50, 75, 90, 100]

# Canvas settings
canvas_height = 750
canvas_width = 1300
top_bar_height = 80
icon_size = 60

# --- NEW SETTINGS ---
SELECTION_HOLD_TIME = 0.8  # Reduced from 2.0 for faster selection
UI_ALPHA = 0.7             # Transparency level (0.0 transparent -> 1.0 solid)

# State variables
canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255
current_color = colors[0]
current_pen = "pen"
current_stroke = "solid"
current_thickness = 10

# UI state
color_box_open = False
pen_box_open = False
stroke_box_open = False
shape_box_open = False
size_box_open = False
show_info = False

# Drawing state
prev_point = None
is_drawing = False
history = []
redo_stack = []
last_save_time = 0

# Gesture tracking
thumb_hover_target = None
thumb_hover_start = None

# Hand locking (First Come First Serve)
locked_hand_id = None
hand_lock_position = None

# Thread control
running = False
_thread = None

# Enhanced smoothing - REDUCED BUFFER FOR LESS LAG
smoothing_buffer = deque(maxlen=6) # Reduced from 15 to 6 for responsiveness
SMOOTHING_WEIGHT = 0.6

# Shape drawing
shape_start_point = None
drawing_shape = None

# ---------------- HELPER FUNCTIONS ----------------

def calculate_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def get_hand_identifier(hand_landmarks):
    lm = hand_landmarks.landmark
    palm_x = (lm[0].x + lm[9].x) / 2
    palm_y = (lm[0].y + lm[9].y) / 2
    return (palm_x, palm_y)

def is_same_hand(pos1, pos2, threshold=0.15):
    if pos1 is None or pos2 is None:
        return False
    dist = math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    return dist < threshold

def get_smoothed_point(new_point):
    global smoothing_buffer
    
    if new_point is None:
        return None
    
    smoothing_buffer.append(new_point)
    
    if len(smoothing_buffer) < 2:
        return new_point
    
    # Simple weighted average is faster than complex linear space
    x_avg = int(sum([p[0] for p in smoothing_buffer]) / len(smoothing_buffer))
    y_avg = int(sum([p[1] for p in smoothing_buffer]) / len(smoothing_buffer))
    
    return (x_avg, y_avg)

def detect_thumb_pointing(hand_landmarks, w, h):
    lm = hand_landmarks.landmark
    thumb_tip_x = int(lm[4].x * w)
    thumb_tip_y = int(lm[4].y * h)
    
    thumb_extended = calculate_distance(
        (lm[4].x * w, lm[4].y * h),
        (lm[2].x * w, lm[2].y * h)
    ) > 40
    
    if thumb_extended:
        return (thumb_tip_x, thumb_tip_y)
    return None

def finger_up(lm, tip_idx, pip_idx):
    return lm[tip_idx].y < lm[pip_idx].y

def get_pen_thickness():
    if current_pen == "eraser":
        return current_thickness
    
    thickness_map = {
        "pen": min(current_thickness, 15),
        "ink": max(2, min(current_thickness // 3, 8)),
        "pencil": min(current_thickness, 20),
        "marker": max(current_thickness, 15)
    }
    return thickness_map.get(current_pen, current_thickness)

def draw_styled_line(img, p1, p2, color, thickness, style):
    if p1 is None or p2 is None:
        return
        
    if style == "solid":
        cv2.line(img, p1, p2, color, thickness, lineType=cv2.LINE_AA)
    elif style == "dashed":
        dist = int(np.hypot(p2[0]-p1[0], p2[1]-p1[1]))
        if dist == 0: return
        dash_len, gap = 15, 8
        for start in range(0, dist, dash_len + gap):
            t0 = start / dist
            t1 = min((start + dash_len) / dist, 1.0)
            sx = int(p1[0] + (p2[0] - p1[0]) * t0)
            sy = int(p1[1] + (p2[1] - p1[1]) * t0)
            ex = int(p1[0] + (p2[0] - p1[0]) * t1)
            ey = int(p1[1] + (p2[1] - p1[1]) * t1)
            cv2.line(img, (sx, sy), (ex, ey), color, thickness, lineType=cv2.LINE_AA)
    elif style == "dotted":
        dist = int(np.hypot(p2[0]-p1[0], p2[1]-p1[1]))
        if dist == 0: return
        step = max(8, thickness)
        for i in range(0, dist, step):
            t = i / dist
            x = int(p1[0] + (p2[0] - p1[0]) * t)
            y = int(p1[1] + (p2[1] - p1[1]) * t)
            cv2.circle(img, (x, y), max(2, thickness//2), color, -1, lineType=cv2.LINE_AA)
    elif style == "zigzag":
        mx = (p1[0] + p2[0]) // 2
        my = (p1[1] + p2[1]) // 2
        offset = max(8, thickness * 2)
        cv2.line(img, p1, (mx, my - offset), color, thickness, lineType=cv2.LINE_AA)
        cv2.line(img, (mx, my - offset), p2, color, thickness, lineType=cv2.LINE_AA)

def draw_shape(img, shape_type, start, end, color, thickness):
    if shape_type == "circle":
        center = ((start[0] + end[0]) // 2, (start[1] + end[1]) // 2)
        radius = int(calculate_distance(start, end) / 2)
        cv2.circle(img, center, radius, color, thickness, lineType=cv2.LINE_AA)
    elif shape_type == "rectangle":
        cv2.rectangle(img, start, end, color, thickness, lineType=cv2.LINE_AA)
    elif shape_type == "square":
        side = max(abs(end[0] - start[0]), abs(end[1] - start[1]))
        end_sq = (start[0] + side, start[1] + side)
        cv2.rectangle(img, start, end_sq, color, thickness, lineType=cv2.LINE_AA)
    elif shape_type == "triangle":
        top = (start[0], end[1])
        bottom_left = (start[0] - abs(end[0] - start[0]), start[1])
        bottom_right = (start[0] + abs(end[0] - start[0]), start[1])
        pts = np.array([top, bottom_left, bottom_right], np.int32)
        cv2.polylines(img, [pts], True, color, thickness, lineType=cv2.LINE_AA)

# ---------------- UI DRAWING FUNCTIONS ----------------

def draw_transparent_rect(img, top_left, bottom_right, color, alpha):
    """Helper to draw semi-transparent rectangles."""
    overlay = img.copy()
    cv2.rectangle(overlay, top_left, bottom_right, color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

def draw_status_bar(frame):
    """Draws a fixed status bar at the bottom showing current settings."""
    bar_height = 50
    y_start = canvas_height - bar_height
    
    # Draw transparent background
    draw_transparent_rect(frame, (0, y_start), (canvas_width, canvas_height), (20, 20, 20), 0.8)
    
    # Info Text
    info_text = f"TOOL: {current_pen.upper()}  |  SIZE: {current_thickness}  |  STYLE: {current_stroke.upper()}  |  SHAPE: {str(drawing_shape).upper() if drawing_shape else 'NONE'}"
    cv2.putText(frame, info_text, (80, y_start + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, lineType=cv2.LINE_AA)
    
    # Color Swatch (Left side)
    cv2.circle(frame, (40, y_start + 25), 18, current_color, -1)
    cv2.circle(frame, (40, y_start + 25), 18, (255, 255, 255), 2) # White border

def draw_icon_button(frame, x, y, icon_type, hover=False):
    bg_color = (220, 220, 220) if not hover else (180, 180, 255)
    # Using direct rectangle for buttons is fine (no need for alpha here for crispness), 
    # but let's make hover smoother
    
    cv2.rectangle(frame, (x, y), (x + icon_size, y + icon_size), bg_color, -1, lineType=cv2.LINE_AA)
    cv2.rectangle(frame, (x, y), (x + icon_size, y + icon_size), (100, 100, 100), 2, lineType=cv2.LINE_AA)
    
    center_x, center_y = x + icon_size // 2, y + icon_size // 2
    
    if icon_type == "colors":
        cv2.circle(frame, (center_x - 10, center_y - 10), 6, (255, 0, 0), -1)
        cv2.circle(frame, (center_x + 10, center_y - 10), 6, (0, 255, 0), -1)
        cv2.circle(frame, (center_x, center_y + 10), 6, (0, 0, 255), -1)
    elif icon_type == "pen":
        cv2.line(frame, (center_x - 15, center_y + 15), (center_x + 15, center_y - 15), (0, 0, 0), 3)
        cv2.circle(frame, (center_x + 15, center_y - 15), 5, (0, 0, 0), -1)
    elif icon_type == "stroke":
        cv2.line(frame, (center_x - 15, center_y), (center_x + 15, center_y), (0, 0, 0), 2)
        cv2.line(frame, (center_x - 15, center_y + 10), (center_x + 15, center_y + 10), (0, 0, 0), 2)
    elif icon_type == "size":
        cv2.circle(frame, (center_x, center_y - 8), 3, (0, 0, 0), -1)
        cv2.circle(frame, (center_x, center_y), 6, (0, 0, 0), -1)
        cv2.circle(frame, (center_x, center_y + 10), 9, (0, 0, 0), -1)
    elif icon_type == "save":
        cv2.rectangle(frame, (center_x - 12, center_y - 15), (center_x + 12, center_y + 15), (0, 0, 0), 2)
        cv2.rectangle(frame, (center_x - 8, center_y + 5), (center_x + 8, center_y + 15), (0, 0, 0), -1)
    elif icon_type == "info":
        cv2.circle(frame, (center_x, center_y), 15, (0, 0, 0), 2)
        cv2.putText(frame, "i", (center_x - 5, center_y + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    elif icon_type == "shapes":
        cv2.circle(frame, (center_x, center_y - 5), 8, (0, 0, 0), 2)
        cv2.rectangle(frame, (center_x - 8, center_y + 5), (center_x + 8, center_y + 15), (0, 0, 0), 2)
    elif icon_type == "clear":
        cv2.rectangle(frame, (center_x - 10, center_y - 5), (center_x + 10, center_y + 15), (0, 0, 0), 2)
        cv2.line(frame, (center_x - 12, center_y - 8), (center_x + 12, center_y - 8), (0, 0, 0), 2)

def draw_top_bar(frame, thumb_pos=None):
    cv2.rectangle(frame, (0, 0), (canvas_width, top_bar_height), (50, 50, 50), -1)
    
    left_icons = [("colors", 10), ("pen", 80), ("size", 150), ("stroke", 220)]
    for icon_type, x_pos in left_icons:
        hover = thumb_pos and (x_pos <= thumb_pos[0] <= x_pos + icon_size and 10 <= thumb_pos[1] <= 10 + icon_size)
        draw_icon_button(frame, x_pos, 10, icon_type, hover)
    
    undo_x, redo_x = 450, 530
    undo_hover = thumb_pos and (undo_x <= thumb_pos[0] <= undo_x + 60 and 10 <= thumb_pos[1] <= 70)
    redo_hover = thumb_pos and (redo_x <= thumb_pos[0] <= redo_x + 60 and 10 <= thumb_pos[1] <= 70)
    
    undo_color = (100, 150, 255) if undo_hover else (80, 120, 200)
    cv2.rectangle(frame, (undo_x, 10), (undo_x + 60, 70), undo_color, -1)
    cv2.rectangle(frame, (undo_x, 10), (undo_x + 60, 70), (255, 255, 255), 2)
    cv2.arrowedLine(frame, (undo_x + 45, 40), (undo_x + 15, 40), (255, 255, 255), 3, tipLength=0.4)
    
    redo_color = (100, 150, 255) if redo_hover else (80, 120, 200)
    cv2.rectangle(frame, (redo_x, 10), (redo_x + 60, 70), redo_color, -1)
    cv2.rectangle(frame, (redo_x, 10), (redo_x + 60, 70), (255, 255, 255), 2)
    cv2.arrowedLine(frame, (redo_x + 15, 40), (redo_x + 45, 40), (255, 255, 255), 3, tipLength=0.4)
    
    right_icons = [("clear", canvas_width - 70), ("shapes", canvas_width - 140), ("info", canvas_width - 210), ("save", canvas_width - 280)]
    for icon_type, x_pos in right_icons:
        hover = thumb_pos and (x_pos <= thumb_pos[0] <= x_pos + icon_size and 10 <= thumb_pos[1] <= 10 + icon_size)
        draw_icon_button(frame, x_pos, 10, icon_type, hover)

def draw_color_box(frame, thumb_pos=None):
    if not color_box_open: return
    
    box_x, box_y = 10, top_bar_height + 10
    box_width, box_height = 250, 450
    
    # Transparent Background
    draw_transparent_rect(frame, (box_x, box_y), (box_x + box_width, box_y + box_height), (240, 240, 240), UI_ALPHA)
    cv2.rectangle(frame, (box_x, box_y), (box_x + box_width, box_y + box_height), (100, 100, 100), 2)
    
    cv2.putText(frame, "COLORS", (box_x + 10, box_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    swatch_size = 45
    cols = 2
    for i, (color, name) in enumerate(zip(colors, color_names)):
        row = i // cols
        col = i % cols
        x = box_x + 20 + col * (swatch_size + 70)
        y = box_y + 60 + row * (swatch_size + 15)
        
        hover = thumb_pos and (x <= thumb_pos[0] <= x + swatch_size and y <= thumb_pos[1] <= y + swatch_size)
        
        cv2.rectangle(frame, (x, y), (x + swatch_size, y + swatch_size), color, -1)
        if color == current_color:
            cv2.rectangle(frame, (x - 3, y - 3), (x + swatch_size + 3, y + swatch_size + 3), (255, 215, 0), 4)
        elif hover:
            cv2.rectangle(frame, (x - 2, y - 2), (x + swatch_size + 2, y + swatch_size + 2), (0, 255, 0), 3)
        else:
            cv2.rectangle(frame, (x, y), (x + swatch_size, y + swatch_size), (100, 100, 100), 2)
        
        cv2.putText(frame, name[:6], (x + swatch_size + 5, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

def draw_pen_box(frame, thumb_pos=None):
    if not pen_box_open: return
    
    box_x, box_y = 80, top_bar_height + 10
    box_width, box_height = 200, 350
    
    draw_transparent_rect(frame, (box_x, box_y), (box_x + box_width, box_y + box_height), (240, 240, 240), UI_ALPHA)
    cv2.rectangle(frame, (box_x, box_y), (box_x + box_width, box_y + box_height), (100, 100, 100), 2)
    
    cv2.putText(frame, "PEN TYPES", (box_x + 10, box_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    for i, pen_name in enumerate(pen_types):
        y = box_y + 60 + i * 55
        x = box_x + 20
        hover = thumb_pos and (x <= thumb_pos[0] <= x + 160 and y - 15 <= thumb_pos[1] <= y + 30)
        
        bg_color = (200, 255, 200) if pen_name == current_pen else ((220, 220, 255) if hover else (255, 255, 255))
        cv2.rectangle(frame, (x, y - 15), (x + 160, y + 30), bg_color, -1)
        cv2.rectangle(frame, (x, y - 15), (x + 160, y + 30), (100, 100, 100), 2)
        cv2.putText(frame, pen_name.upper(), (x + 10, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

def draw_size_box(frame, thumb_pos=None):
    if not size_box_open: return
    
    box_x, box_y = 150, top_bar_height + 10
    box_width, box_height = 200, 450
    
    draw_transparent_rect(frame, (box_x, box_y), (box_x + box_width, box_y + box_height), (240, 240, 240), UI_ALPHA)
    cv2.rectangle(frame, (box_x, box_y), (box_x + box_width, box_y + box_height), (100, 100, 100), 2)
    
    cv2.putText(frame, "SIZE", (box_x + 10, box_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    for i, size in enumerate(size_options):
        y = box_y + 60 + i * 55
        x = box_x + 20
        hover = thumb_pos and (x <= thumb_pos[0] <= x + 160 and y - 15 <= thumb_pos[1] <= y + 30)
        
        bg_color = (200, 255, 200) if size == current_thickness else ((220, 220, 255) if hover else (255, 255, 255))
        cv2.rectangle(frame, (x, y - 15), (x + 160, y + 30), bg_color, -1)
        cv2.rectangle(frame, (x, y - 15), (x + 160, y + 30), (100, 100, 100), 2)
        cv2.putText(frame, str(size), (x + 10, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        circle_size = min(size // 4, 15)
        cv2.circle(frame, (x + 140, y + 5), max(2, circle_size), (100, 100, 100), -1)

def draw_stroke_box(frame, thumb_pos=None):
    if not stroke_box_open: return
    
    box_x, box_y = 220, top_bar_height + 10
    box_width, box_height = 200, 300
    
    draw_transparent_rect(frame, (box_x, box_y), (box_x + box_width, box_y + box_height), (240, 240, 240), UI_ALPHA)
    cv2.rectangle(frame, (box_x, box_y), (box_x + box_width, box_y + box_height), (100, 100, 100), 2)
    
    cv2.putText(frame, "STROKES", (box_x + 10, box_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    for i, stroke_name in enumerate(stroke_styles):
        y = box_y + 60 + i * 55
        x = box_x + 20
        hover = thumb_pos and (x <= thumb_pos[0] <= x + 160 and y - 15 <= thumb_pos[1] <= y + 30)
        
        bg_color = (200, 255, 200) if stroke_name == current_stroke else ((220, 220, 255) if hover else (255, 255, 255))
        cv2.rectangle(frame, (x, y - 15), (x + 160, y + 30), bg_color, -1)
        cv2.rectangle(frame, (x, y - 15), (x + 160, y + 30), (100, 100, 100), 2)
        cv2.putText(frame, stroke_name.upper(), (x + 10, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

def draw_shape_box(frame, thumb_pos=None):
    if not shape_box_open: return
    
    box_x, box_y = canvas_width - 360, top_bar_height + 10
    box_width, box_height = 200, 300
    
    draw_transparent_rect(frame, (box_x, box_y), (box_x + box_width, box_y + box_height), (240, 240, 240), UI_ALPHA)
    cv2.rectangle(frame, (box_x, box_y), (box_x + box_width, box_y + box_height), (100, 100, 100), 2)
    
    cv2.putText(frame, "SHAPES", (box_x + 10, box_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    for i, shape_name in enumerate(shapes):
        y = box_y + 60 + i * 55
        x = box_x + 20
        hover = thumb_pos and (x <= thumb_pos[0] <= x + 160 and y - 15 <= thumb_pos[1] <= y + 30)
        
        bg_color = (220, 220, 255) if hover else (255, 255, 255)
        cv2.rectangle(frame, (x, y - 15), (x + 160, y + 30), bg_color, -1)
        cv2.rectangle(frame, (x, y - 15), (x + 160, y + 30), (100, 100, 100), 2)
        cv2.putText(frame, shape_name.upper(), (x + 10, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

def draw_info_panel(frame):
    if not show_info: return
    
    panel_x, panel_y = canvas_width - 420, top_bar_height + 10
    panel_width, panel_height = 400, 480
    
    draw_transparent_rect(frame, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height), (30, 30, 30), 0.8)
    cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height), (255, 255, 255), 2)
    
    cv2.putText(frame, "CONTROLS & INFO", (panel_x + 10, panel_y + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    info_lines = [
        "",
        "HAND LOCK (First Come First Serve):",
        "- First detected hand gets control",
        "- Other hands are ignored",
        "- Remove hand to unlock",
        "",
        "DRAWING:",
        "- Index finger up: Draw on canvas",
        "- Index + middle up: Navigate only",
        "",
        f"THUMB POINTING (Hold {SELECTION_HOLD_TIME}s):",
        "- Point thumb at ANY option to select",
        "- Works for ALL buttons and menus:",
        "  * Colors, Pen types, Sizes",
        "  * Stroke styles, Shapes",
        "  * Undo, Redo, Save, Clear, Info",
        "- Green progress bar shows countdown",
        "",
        "KEYBOARD:",
        "- Q: Exit canvas",
    ]
    
    y_offset = panel_y + 60
    for line in info_lines:
        cv2.putText(frame, line, (panel_x + 15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, lineType=cv2.LINE_AA)
        y_offset += 21

# ---------------- MAIN LOOP ----------------

def _detect_loop():
    global running, prev_point, canvas, current_color, current_pen, current_stroke
    global color_box_open, pen_box_open, stroke_box_open, shape_box_open, show_info, size_box_open
    global is_drawing, history, redo_stack, current_thickness
    global thumb_hover_target, thumb_hover_start
    global shape_start_point, drawing_shape
    global locked_hand_id, hand_lock_position

    with mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.75) as hands:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Could not open webcam")
            running = False
            return

        cap.set(3, canvas_width)
        cap.set(4, canvas_height)

        while running:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            frame = cv2.resize(frame, (canvas_width, canvas_height))

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            thumb_pos = None
            current_time = time.time()
            active_hand = None

            if results.multi_hand_landmarks:
                # HAND LOCKING LOGIC
                if locked_hand_id is None:
                    active_hand = results.multi_hand_landmarks[0]
                    hand_lock_position = get_hand_identifier(active_hand)
                    locked_hand_id = 0
                else:
                    for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                        current_hand_pos = get_hand_identifier(hand_landmarks)
                        if is_same_hand(hand_lock_position, current_hand_pos):
                            active_hand = hand_landmarks
                            hand_lock_position = current_hand_pos
                            break
                    else:
                        locked_hand_id = None
                        hand_lock_position = None
                        smoothing_buffer.clear()
                        is_drawing = False
                        prev_point = None

                # Process active hand
                if active_hand:
                    mp_drawing.draw_landmarks(frame, active_hand, mp_hands.HAND_CONNECTIONS)

                    lm = active_hand.landmark
                    w, h = frame.shape[1], frame.shape[0]
                    
                    thumb_pos = detect_thumb_pointing(active_hand, w, h)
                    
                    ix, iy = int(lm[8].x * w), int(lm[8].y * h)
                    raw_fingertip = (ix, iy)
                    fingertip = get_smoothed_point(raw_fingertip)

                    index_up = finger_up(lm, 8, 6)
                    middle_up = finger_up(lm, 12, 10)

                    # THUMB SELECTION LOGIC
                    if thumb_pos:
                        selection_made = False
                        
                        top_bar_buttons = [
                            ("colors_icon", 10, 10, 70, 70, lambda: toggle_menu("colors")),
                            ("pen_icon", 80, 10, 140, 70, lambda: toggle_menu("pen")),
                            ("size_icon", 150, 10, 210, 70, lambda: toggle_menu("size")),
                            ("stroke_icon", 220, 10, 280, 70, lambda: toggle_menu("stroke")),
                            ("undo", 450, 10, 510, 70, lambda: undo_action()),
                            ("redo", 530, 10, 590, 70, lambda: redo_action()),
                            ("save", canvas_width - 280, 10, canvas_width - 220, 70, lambda: save_canvas()),
                            ("info", canvas_width - 210, 10, canvas_width - 150, 70, lambda: toggle_info()),
                            ("shapes", canvas_width - 140, 10, canvas_width - 80, 70, lambda: toggle_menu("shapes")),
                            ("clear", canvas_width - 70, 10, canvas_width - 10, 70, lambda: clear_canvas())
                        ]
                        
                        for btn_name, x1, y1, x2, y2, action in top_bar_buttons:
                            if x1 <= thumb_pos[0] <= x2 and y1 <= thumb_pos[1] <= y2:
                                if thumb_hover_target != btn_name:
                                    thumb_hover_target = btn_name
                                    thumb_hover_start = current_time
                                elif current_time - thumb_hover_start >= SELECTION_HOLD_TIME:
                                    action()
                                    thumb_hover_target = None
                                    thumb_hover_start = None
                                else:
                                    progress = (current_time - thumb_hover_start) / SELECTION_HOLD_TIME
                                    cv2.rectangle(frame, (x1, y2 + 2), (x1 + int((x2 - x1) * progress), y2 + 6), (0, 255, 0), -1)
                                selection_made = True
                                break
                        
                        # --- BOX SELECTIONS (Generalized Logic) ---
                        # To keep code short, define all boxes and iterate
                        # Only check boxes if they are OPEN
                        menus = [
                            (color_box_open, "color", len(colors), 30, top_bar_height+70, 2, 45, 70, 15, lambda i: set_color(i)),
                            (pen_box_open, "pen", len(pen_types), 100, top_bar_height+60, 1, 160, 0, 55, lambda i: set_pen(i)),
                            (size_box_open, "size", len(size_options), 170, top_bar_height+60, 1, 160, 0, 55, lambda i: set_size(i)),
                            (stroke_box_open, "stroke", len(stroke_styles), 240, top_bar_height+60, 1, 160, 0, 55, lambda i: set_stroke(i)),
                            (shape_box_open, "shape", len(shapes), canvas_width-340, top_bar_height+60, 1, 160, 0, 55, lambda i: set_shape(i))
                        ]

                        for isOpen, name, count, startX, startY, cols, w, x_gap, y_gap, action in menus:
                            if not selection_made and isOpen:
                                for i in range(count):
                                    row = i // cols
                                    col = i % cols
                                    x = startX + col * (w + x_gap)
                                    y = startY + row * (45 + y_gap) if name=="color" else startY + i*55 
                                    
                                    h_hit = 45 if name=="color" else 45
                                    
                                    if x <= thumb_pos[0] <= x + w and y <= thumb_pos[1] <= y + h_hit:
                                        target = (name, i)
                                        if thumb_hover_target != target:
                                            thumb_hover_target = target
                                            thumb_hover_start = current_time
                                        elif current_time - thumb_hover_start >= SELECTION_HOLD_TIME:
                                            action(i)
                                            thumb_hover_target = None
                                            thumb_hover_start = None
                                        else:
                                            progress = (current_time - thumb_hover_start) / SELECTION_HOLD_TIME
                                            cv2.rectangle(frame, (x, y + h_hit - 5), (x + int(w * progress), y + h_hit), (0, 255, 0), -1)
                                        selection_made = True
                                        break

                        if not selection_made:
                            thumb_hover_target = None
                            thumb_hover_start = None

                    # DRAWING LOGIC
                    if fingertip and iy > top_bar_height:
                        cx, cy = fingertip
                        
                        if drawing_shape:
                            if index_up and not middle_up:
                                if shape_start_point is None:
                                    shape_start_point = (cx, cy)
                                    history.append(canvas.copy())
                            else:
                                if shape_start_point:
                                    thickness = get_pen_thickness()
                                    draw_shape(canvas, drawing_shape, shape_start_point, (cx, cy), current_color, thickness)
                                    shape_start_point = None
                                    drawing_shape = None
                        else:
                            if index_up and not middle_up:
                                if not is_drawing:
                                    history.append(canvas.copy())
                                    redo_stack.clear()
                                    is_drawing = True
                                    prev_point = (cx, cy)
                                else:
                                    if prev_point:
                                        thickness = get_pen_thickness()
                                        color = (255, 255, 255) if current_pen == "eraser" else current_color
                                        draw_styled_line(canvas, prev_point, (cx, cy), color, thickness, current_stroke)
                                    prev_point = (cx, cy)
                            else:
                                is_drawing = False
                                prev_point = None
            else:
                if locked_hand_id is not None:
                    locked_hand_id = None
                    hand_lock_position = None
                    smoothing_buffer.clear()
                is_drawing = False
                prev_point = None
                thumb_hover_target = None
                thumb_hover_start = None

            if locked_hand_id is not None:
                cv2.putText(frame, "LOCKED", (10, canvas_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Draw UI
            draw_top_bar(frame, thumb_pos)
            draw_color_box(frame, thumb_pos)
            draw_pen_box(frame, thumb_pos)
            draw_size_box(frame, thumb_pos)
            draw_stroke_box(frame, thumb_pos)
            draw_shape_box(frame, thumb_pos)
            draw_info_panel(frame)
            draw_status_bar(frame) # New status bar

            # Blend canvas
            masked = cv2.addWeighted(frame, 0.3, canvas, 0.7, 0)
            
            cv2.imshow("Advanced Drawing Canvas", masked)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                running = False
                break

        cap.release()
        cv2.destroyAllWindows()
        print("✅ Canvas closed successfully")

# Helper functions for actions
def toggle_menu(menu_name):
    global color_box_open, pen_box_open, size_box_open, stroke_box_open, shape_box_open
    
    # Close all others first
    color_box_open = False
    pen_box_open = False
    size_box_open = False
    stroke_box_open = False
    shape_box_open = False
    
    if menu_name == "colors": color_box_open = True
    elif menu_name == "pen": pen_box_open = True
    elif menu_name == "size": size_box_open = True
    elif menu_name == "stroke": stroke_box_open = True
    elif menu_name == "shapes": shape_box_open = True

def undo_action():
    global canvas, redo_stack, history
    if history:
        redo_stack.append(canvas.copy())
        canvas[:] = history.pop()

def redo_action():
    global canvas, redo_stack, history
    if redo_stack:
        history.append(canvas.copy())
        canvas[:] = redo_stack.pop()

def save_canvas():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"canvas_{timestamp}.png"
    cv2.imwrite(filename, canvas)
    print(f"✅ Saved: {filename}")

def toggle_info():
    global show_info
    show_info = not show_info

def clear_canvas():
    global canvas, history, redo_stack
    canvas[:] = 255
    history.clear()
    redo_stack.clear()

def set_color(index):
    global current_color
    current_color = colors[index]

def set_pen(index):
    global current_pen, current_color
    current_pen = pen_types[index]
    if pen_types[index] == "eraser":
        current_color = (255, 255, 255)

def set_size(index):
    global current_thickness
    current_thickness = size_options[index]

def set_stroke(index):
    global current_stroke
    current_stroke = stroke_styles[index]

def set_shape(index):
    global drawing_shape
    drawing_shape = shapes[index]

# ---------------- START/STOP ----------------

def start_canvas():
    global running, _thread
    if running:
        return
    running = True
    _thread = threading.Thread(target=_detect_loop, daemon=True)
    _thread.start()

def stop_canvas():
    global running, _thread
    running = False
    if _thread:
        _thread.join(timeout=2.0)

# ---------------- ENTRY POINT ----------------

if __name__ == "__main__":
    print("🎨 Advanced Drawing Canvas - Thumb Selection & Hand Lock")
    start_canvas()
    try:
        while running:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
    finally:
        stop_canvas()