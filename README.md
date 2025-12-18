# Gesture Gaze Interaction System

A Next-Generation Human-Computer Interaction (HCI) system that enables touchless computer control using **hand gestures** and **eye movements**. Built with Python and Computer Vision, it features a futuristic web interface and an integrated AI Assistant powered by Google Gemini.

-----
<img width="1899" height="870" alt="Image" src="https://github.com/user-attachments/assets/26572b47-d9c5-433b-a6f6-82a0464859e9" />

##  Features

### 🖐️ 1. Hand Gesture Mouse

Control your cursor with precision hand tracking.

  * **Cursor Movement:** Tracks the Index Finger tip with smoothing to reduce jitter.
  * **Left Click:** Pinch Thumb & Index Finger together.
  * **Smart Scrolling:**
      * **Normal Scroll:** Move hand Up/Down relative to baseline position.
      * **Fast Scroll:** Make a **Fist** while moving Up/Down for high-speed scrolling.

### 👁️ 2. Eye Tracking Mouse

Hands-free accessibility mode for users with limited motor function.

  * **Gaze Control:** Cursor follows the center of your Iris.
  * **Click:** Perform a **Double Blink** (both eyes).
  * **Scroll Down:** Wink **Left Eye**.
  * **Scroll Up:** Wink **Right Eye**.

### 🎨 3. Air Canvas (AR Drawing)

A virtual whiteboard for drawing in mid-air.

  * **Draw:** Raise **Index Finger** to draw.
  * **Hover:** Raise **Index + Middle Fingers** to move cursor without drawing.
  * **Tools:**
      * Virtual Toolbar: Clear, Undo, Eraser, Save.
      * Sidebar Palette: Select from 10+ colors.
      * Brush Styles: Normal, Dashed, Zigzag, Straight.

### 🐍 4. Gesture-Controlled Snake Game

A classic game reinvented with computer vision.

  * **Control:** Point your Index Finger (Up, Down, Left, Right) relative to your wrist to steer the snake.
  * **Engine:** Hybrid OpenCV (Input) and Pygame (Rendering).

### 🤖 5. AI Assistant (RAG Chatbot)

  * **Powered by:** Google Gemini 1.5 Flash.
  * **Context Aware:** Uses a custom knowledge base to answer specific questions about the project's controls and features (e.g., *"How do I scroll in Eye Mode?"*).

-----

## 🛠️ Tech Stack

  * **Frontend:** HTML, CSS, JavaScript.
  * **Backend:** Python (Flask).
  * **Computer Vision:** OpenCV, MediaPipe (Hands & FaceMesh).
  * **Automation:** PyAutoGUI (Mouse control).
  * **Game Engine:** Pygame.
  * **Generative AI:** Google Generative AI (Gemini API).

-----

## 📦 Installation

1.  **Clone the Repository**

    ```bash
    git clone https://github.com/your-username/gesture-gaze-interaction-system.git
    cd gesture-gaze-interaction-system
    ```

2.  **Install Dependencies**

    ```bash
    pip install -r requirements.txt
    ```

-----

## 🎮 Usage Guide

### Starting the App

Run the Flask application:

```bash
python app.py
```

Open your browser and go to: `http://127.0.0.1:5000/`

### 🕹️ Controls Reference

| Mode | Gesture / Action | Result |
| :--- | :--- | :--- |
| **Hand Mouse** | Index Finger Move | Move Cursor |
| | Pinch (Thumb+Index) | Click / Drag |
| | Hand Up/Down | Scroll Page |
| | Fist + Up/Down | Fast Scroll |
| **Eye Mouse** | Iris Movement | Move Cursor |
| | Double Blink | Left Click |
| | Left Wink | Scroll Down |
| | Right Wink | Scroll Up |
| **Canvas** | Index Finger Up | Draw Line |
| | Index + Middle Up | Hover (No Draw) |
| | Keyboard 'C' | Clear Canvas |
| | Keyboard 'A' | Save Image |
| **Snake Game** | Point Finger | Change Direction |

-----

## 📂 Project Structure

```text
gesture-gaze-interaction-system/
│
├── app.py                   # Main Flask Application
├── config.py                # API Key Configuration
├── requirements.txt         # Python Dependencies
│
├── modules/
│   ├── hand_gesture_mouse.py  # Hand Tracking Logic
│   ├── eye_tracking_mouse.py  # Eye Tracking Logic
│   ├── drawing_canvas.py      # Virtual Painting Logic
│   ├── game.py                # Snake Game Logic
│   └── rag_chatbot.py         # AI Chatbot Logic
│
├── knowledge_base/
│   └── hand_eye_mouse_info.txt # RAG Context Data
│
├── static/
│   ├── styles.css           # UI Styling
│   └── images/              # Backgrounds & Assets
│
└── templates/
    └── index.html           # Main Dashboard
```

-----

## ⚠️ Troubleshooting

  * **Camera Not Opening:** Ensure no other app is using the webcam.
  * **Scrolling Issues:** Adjust the lighting. The system needs clear visibility of your fingers to calculate depth and gestures.

-----
