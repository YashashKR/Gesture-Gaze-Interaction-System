import sys
import time
import threading
import math
import numpy as np
import pygame
import os

try:
    import cv2
    import mediapipe as mp
    mp_hands = mp.solutions.hands
except ImportError as e:
    print(f"ERROR: MediaPipe/OpenCV not available: {e}")
    sys.exit(1)

gesture_direction = None
gesture_lock = threading.Lock()
STOP_THREADS = False
DEADZONE = 0.12
FPS_CAM = 10

def camera_worker(camera_index=0):
    global gesture_direction, STOP_THREADS
    
    try:
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            return
    except Exception as e:
        return

    with mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    ) as hands:
        prev_time = 0
        while not STOP_THREADS:
            t0 = time.time()
            ret, frame = cap.read()
            if not ret:
                break
            # flip for natural mirror control
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            dir_now = None
            if results.multi_hand_landmarks:
                hand = results.multi_hand_landmarks[0]
                lm = hand.landmark
                # wrist (0) and index fingertip (8)
                wrist = lm[0]
                index_tip = lm[8]

                dx = index_tip.x - wrist.x
                dy = index_tip.y - wrist.y  # y grows downward in image coordinates

                # Convert to normalized distances
                # Use absolute values relative to a threshold
                abs_dx = abs(dx)
                abs_dy = abs(dy)

                # If movement is too small, ignore (deadzone)
                if math.hypot(dx, dy) > DEADZONE:
                    # Decide dominant direction
                    if abs_dx > abs_dy:
                        dir_now = "RIGHT" if dx > 0 else "LEFT"
                    else:
                        # dy positive -> index is below wrist (pointing down on camera),
                        # but we consider that as DOWN. Note camera y points down.
                        dir_now = "DOWN" if dy > 0 else "UP"
                else:
                    dir_now = None

                # Optionally show a small debug window with landmarks (comment out to reduce CPU)
                # mp.solutions.drawing_utils.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

            # update shared variable
            with gesture_lock:
                gesture_direction = dir_now

            cv2.putText(frame, f"Gesture: {dir_now}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, "Press 'q' to close", (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            try:
                cv2.imshow("Gesture Control", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            except:
                pass

            time.sleep(1.0 / FPS_CAM)

    try:
        cap.release()
        cv2.destroyAllWindows()
    except:
        pass


SCREEN_W, SCREEN_H = 640, 480
CELL_SIZE = 20
COLS = SCREEN_W // CELL_SIZE
ROWS = SCREEN_H // CELL_SIZE

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 200, 0)
RED = (220, 20, 20)
BLUE = (0, 120, 255)



def run_game():
    global gesture_direction, STOP_THREADS
    
    os.environ['SDL_VIDEODRIVER'] = 'windib'
    
    try:
        pygame.init()
        screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
        pygame.display.set_caption("Gesture Snake Game")
        clock = pygame.time.Clock()
        font = pygame.font.SysFont(None, 28)
        title_font = pygame.font.SysFont(None, 36)
    except Exception as e:
        return

    snake = [(COLS // 2 - i, ROWS // 2) for i in range(3)]
    dir_x, dir_y = 1, 0
    current_dir = "RIGHT"
    
    def spawn_food():
        while True:
            fx = np.random.randint(0, COLS)
            fy = np.random.randint(0, ROWS)
            if (fx, fy) not in snake:
                return (fx, fy)
    
    food = spawn_food()
    score = 0
    move_timer = 0
    move_delay = 10

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_UP and current_dir != "DOWN":
                    current_dir = "UP"
                elif event.key == pygame.K_DOWN and current_dir != "UP":
                    current_dir = "DOWN"
                elif event.key == pygame.K_LEFT and current_dir != "RIGHT":
                    current_dir = "LEFT"
                elif event.key == pygame.K_RIGHT and current_dir != "LEFT":
                    current_dir = "RIGHT"
        
        with gesture_lock:
            g = gesture_direction
        
        if g and g == "UP" and current_dir != "DOWN":
            current_dir = "UP"
        elif g and g == "DOWN" and current_dir != "UP":
            current_dir = "DOWN"
        elif g and g == "LEFT" and current_dir != "RIGHT":
            current_dir = "LEFT"
        elif g and g == "RIGHT" and current_dir != "LEFT":
            current_dir = "RIGHT"
        
        move_timer += 1
        if move_timer >= move_delay:
            move_timer = 0
            
            if current_dir == "UP":
                dir_x, dir_y = 0, -1
            elif current_dir == "DOWN":
                dir_x, dir_y = 0, 1
            elif current_dir == "LEFT":
                dir_x, dir_y = -1, 0
            elif current_dir == "RIGHT":
                dir_x, dir_y = 1, 0
            
            head_x, head_y = snake[0]
            new_head = ((head_x + dir_x) % COLS, (head_y + dir_y) % ROWS)
            
            if new_head in snake:
                running = False
                continue
            
            snake.insert(0, new_head)
            
            if new_head == food:
                score += 1
                food = spawn_food()
            else:
                snake.pop()

        screen.fill(BLACK)
        
        fx, fy = food
        pygame.draw.rect(screen, RED, (fx * CELL_SIZE, fy * CELL_SIZE, CELL_SIZE - 1, CELL_SIZE - 1))
        
        for i, (sx, sy) in enumerate(snake):
            color = BLUE if i == 0 else GREEN
            pygame.draw.rect(screen, color, (sx * CELL_SIZE, sy * CELL_SIZE, CELL_SIZE - 1, CELL_SIZE - 1))
        
        score_text = font.render(f"Score: {score}", True, WHITE)
        gesture_text = font.render(f"Gesture: {gesture_direction or 'None'}", True, WHITE)
        
        screen.blit(score_text, (10, 10))
        screen.blit(gesture_text, (10, 40))
        
        pygame.display.flip()
        clock.tick(60)

    screen.fill(BLACK)
    game_over_text = title_font.render("GAME OVER!", True, WHITE)
    score_text = font.render(f"Final Score: {score}", True, WHITE)
    
    screen.blit(game_over_text, (SCREEN_W//2 - 100, SCREEN_H//2 - 40))
    screen.blit(score_text, (SCREEN_W//2 - 80, SCREEN_H//2))
    pygame.display.flip()
    
    time.sleep(3)
    
    try:
        pygame.quit()
    except:
        pass
    
    STOP_THREADS = True


# ------------------ Main ------------------
if __name__ == "__main__":
    STOP_THREADS = False
    cam_thread = threading.Thread(target=camera_worker, args=(0,), daemon=True)
    cam_thread.start()
    time.sleep(1)
    
    try:
        run_game()
    except:
        pass
    finally:
        STOP_THREADS = True