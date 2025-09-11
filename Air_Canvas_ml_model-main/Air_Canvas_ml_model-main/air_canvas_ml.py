# drawing_canvas_hand.py
# Refined drawing code — separate webcam and canvas, rule: index=draw, index+middle=pause
import cv2
import numpy as np
import mediapipe as mp
import math

# -------------------- configuration --------------------
CANVAS_W, CANVAS_H = 640, 480
BRUSH_THICKNESSES = { "1":5, "2":10, "3":15, "4":20 }
brush_thickness = 5

# Colors (BGR)
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
colorIndex = 0
eraser_mode_manual = False
# -------------------------------------------------------

# Prepare paint window (white)
paintWindow = np.ones((CANVAS_H, CANVAS_W, 3), dtype=np.uint8) * 255

# Button bar (same as before)
def draw_button_bar(img):
    cv2.rectangle(img, (40,1), (140,65), (0,0,0), 2)
    cv2.rectangle(img, (160,1), (255,65), (255,0,0), 2)
    cv2.rectangle(img, (275,1), (370,65), (0,255,0), 2)
    cv2.rectangle(img, (390,1), (485,65), (0,0,255), 2)
    cv2.rectangle(img, (505,1), (600,65), (0,0,0), 2)

    cv2.putText(img, "CLEAR", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    cv2.putText(img, "BLUE", (185, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    cv2.putText(img, "GREEN", (298, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.putText(img, "RED", (420, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(img, "ERASER", (520, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

# mediapipe init
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.75, min_tracking_confidence=0.75)
mpDraw = mp.solutions.drawing_utils

# finger state check (True if up)
def finger_up(lm, tip, pip):
    return lm[tip].y < lm[pip].y  # tip above pip → finger is up

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam")

prev_center = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame_h, frame_w = frame.shape[:2]

    frame_top = frame.copy()
    draw_button_bar(frame_top)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            mpDraw.draw_landmarks(frame_top, handLms, mpHands.HAND_CONNECTIONS)

            lm = handLms.landmark
            ix, iy = int(lm[8].x * frame_w), int(lm[8].y * frame_h)
            mx, my = int(lm[12].x * frame_w), int(lm[12].y * frame_h)

            index_up = finger_up(lm, 8, 6)
            middle_up = finger_up(lm, 12, 10)

            # top bar controls
            if iy <= 65:
                if 40 <= ix <= 140:  # clear
                    paintWindow[67:,:,:] = 255
                elif 160 <= ix <= 255:  # blue
                    colorIndex = 0; eraser_mode_manual = False
                elif 275 <= ix <= 370:  # green
                    colorIndex = 1; eraser_mode_manual = False
                elif 390 <= ix <= 485:  # red
                    colorIndex = 2; eraser_mode_manual = False
                elif 505 <= ix <= 600:  # eraser
                    eraser_mode_manual = True
            else:
                # drawing area
                cx, cy = int(ix * (CANVAS_W / frame_w)), int(iy * (CANVAS_H / frame_h))

                if eraser_mode_manual:
                    cv2.circle(paintWindow, (cx, cy), 50, (255,255,255), -1)
                    prev_center = None
                elif index_up and not middle_up:  # draw only if index up & middle down
                    if prev_center is None:
                        prev_center = (cx, cy)
                    if colorIndex == 0:
                        cv2.line(paintWindow, prev_center, (cx, cy), colors[0], brush_thickness)
                    elif colorIndex == 1:
                        cv2.line(paintWindow, prev_center, (cx, cy), colors[1], brush_thickness)
                    elif colorIndex == 2:
                        cv2.line(paintWindow, prev_center, (cx, cy), colors[2], brush_thickness)
                    prev_center = (cx, cy)
                else:
                    prev_center = None
    else:
        prev_center = None

    # show separate windows
    cv2.imshow("Webcam Feed", frame_top)
    cv2.imshow("Paint (Canvas only)", paintWindow)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        paintWindow[:] = 255
    elif chr(key) in BRUSH_THICKNESSES:
        brush_thickness = BRUSH_THICKNESSES[chr(key)]
        print(f"Brush thickness set to {brush_thickness}")

cap.release()
cv2.destroyAllWindows()
