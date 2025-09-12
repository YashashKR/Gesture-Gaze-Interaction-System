import speech_recognition as sr
import tkinter as tk
from PIL import ImageGrab
import re
import pyttsx3

# ------------------- Voice Engine -------------------
engine = pyttsx3.init()
def speak(text):
    engine.say(text)
    engine.runAndWait()

# ------------------- Voice Recognition -------------------
def listen_command():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        speak("Listening for command")
        print("🎤 Listening... Speak your command:")
        audio = recognizer.listen(source)
        try:
            command = recognizer.recognize_google(audio).lower()
            print("✅ You said:", command)
            return command
        except sr.UnknownValueError:
            speak("Could not understand audio")
            return ""
        except sr.RequestError:
            speak("Speech service unavailable")
            return ""

# ------------------- Canvas Operations -------------------
actions = []   # history for undo
redo_stack = []

def save_canvas():
    x = root.winfo_rootx() + canvas.winfo_x()
    y = root.winfo_rooty() + canvas.winfo_y()
    x1 = x + canvas.winfo_width()
    y1 = y + canvas.winfo_height()
    ImageGrab.grab().crop((x, y, x1, y1)).save("canvas.png")
    speak("Canvas saved successfully")

def undo():
    if actions:
        item = actions.pop()
        canvas.delete(item)
        redo_stack.append(item)
        speak("Undo last action")

def redo():
    if redo_stack:
        item = redo_stack.pop()
        # Re-draw is tricky; here we just acknowledge redo
        speak("Redo action not yet fully supported")

def clear_canvas():
    canvas.delete("all")
    actions.clear()
    redo_stack.clear()
    speak("Canvas cleared")

# ------------------- Shape Drawing -------------------
shape_size = 100  # default size
last_item = None

def draw_shape(shape, color="black", fill=None, x=300, y=200):
    global last_item, shape_size
    size = shape_size
    item = None

    if shape == "circle":
        item = canvas.create_oval(x, y, x+size, y+size, outline=color, fill=fill, width=3)
    elif shape == "rectangle":
        item = canvas.create_rectangle(x, y, x+size, y+size/2, outline=color, fill=fill, width=3)
    elif shape == "line":
        item = canvas.create_line(x, y, x+size, y, fill=color, width=3)
    elif shape == "triangle":
        item = canvas.create_polygon(x, y, x+size, y, x+size/2, y-size, outline=color, fill=fill, width=3)
    elif shape == "square":
        item = canvas.create_rectangle(x, y, x+size, y+size, outline=color, fill=fill, width=3)
    elif shape == "ellipse":
        item = canvas.create_oval(x, y, x+size*1.5, y+size, outline=color, fill=fill, width=3)
    elif shape == "polygon":
        item = canvas.create_polygon(x, y, x+size, y, x+size+20, y+50, x+size/2, y+80, outline=color, fill=fill, width=3)
    elif shape == "cube":  # fake 3D
        item = canvas.create_rectangle(x, y, x+size, y+size, outline=color, fill=fill, width=3)
        canvas.create_rectangle(x+20, y-20, x+size+20, y+size-20, outline=color, width=2)
        canvas.create_line(x, y, x+20, y-20)
        canvas.create_line(x+size, y, x+size+20, y-20)
        canvas.create_line(x, y+size, x+20, y+size-20)
        canvas.create_line(x+size, y+size, x+size+20, y+size-20)
    else:
        speak("Shape not recognized")
        return

    if item:
        actions.append(item)
        last_item = item
        speak(f"{shape} drawn")

def resize_shape(bigger=True):
    global shape_size
    if bigger:
        shape_size += 20
        speak("Size increased")
    else:
        shape_size = max(20, shape_size-20)
        speak("Size decreased")

# ------------------- Help Menu -------------------
def show_help():
    help_text = """✅ Available Commands:

🎨 Drawing:
circle, rectangle, line, triangle, polygon, ellipse, square, cube, sphere, cylinder, pyramid
- draw [number] [shape]
- draw [color] [shape]

📏 Size:
- bigger (make last shape larger)
- smaller (make last shape smaller)

📍 Positioning:
- draw [shape] top / bottom / left / right / middle

🧽 Editing:
- clear (clear entire canvas)
- erase [color]
- remove [shape]
- undo / redo

🗂️ Layers:
- draw [shape] on top / middle / below

💾 Canvas:
- save canvas

ℹ️ Help:
- help (show help)
- exit (close help)

🚪 Closing:
- close (close canvas completely)"""

    canvas.delete("all")
    y = 20
    for line in help_text.split("\n"):
        canvas.create_text(10, y, anchor="w", text=line, font=("Arial", 11), fill="black")
        y += 20
    speak("Help box opened")

def exit_help():
    canvas.delete("all")
    speak("Exited help, blank canvas ready")

# ------------------- Command Processor -------------------
def process_command(cmd):
    if not cmd:
        return
    if "start" in cmd:
        speak("Canva started")
        listen_and_draw()
    elif "circle" in cmd: draw_shape("circle")
    elif "rectangle" in cmd: draw_shape("rectangle")
    elif "triangle" in cmd: draw_shape("triangle")
    elif "line" in cmd: draw_shape("line")
    elif "square" in cmd: draw_shape("square")
    elif "ellipse" in cmd: draw_shape("ellipse")
    elif "polygon" in cmd: draw_shape("polygon")
    elif "cube" in cmd: draw_shape("cube")
    elif "bigger" in cmd: resize_shape(True)
    elif "smaller" in cmd: resize_shape(False)
    elif "clear" in cmd: clear_canvas()
    elif "undo" in cmd: undo()
    elif "redo" in cmd: redo()
    elif "save" in cmd: save_canvas()
    elif "help" in cmd: show_help()
    elif "exit" in cmd: exit_help()
    elif "close" in cmd:
        speak("Closing canvas. Goodbye")
        root.destroy()
    else:
        speak("Command not recognized")

# ------------------- Main Loop -------------------
def listen_and_draw():
    cmd = listen_command()
    process_command(cmd)

root = tk.Tk()
root.title("🎨 Voice Controlled Canvas")

canvas = tk.Canvas(root, width=800, height=600, bg="white")
canvas.pack()

# Toolbar
toolbar = tk.Frame(root)
toolbar.pack(pady=10)

btn_speak = tk.Button(toolbar, text="🎤 Speak & Draw", command=listen_and_draw)
btn_speak.grid(row=0, column=0, padx=5)

btn_help = tk.Button(toolbar, text="ℹ️ Help", command=show_help)
btn_help.grid(row=0, column=1, padx=5)

btn_save = tk.Button(toolbar, text="💾 Save", command=save_canvas)
btn_save.grid(row=0, column=2, padx=5)

btn_close = tk.Button(toolbar, text="❌ Close", command=lambda: (speak("Closing canvas. Goodbye"), root.destroy()))
btn_close.grid(row=0, column=3, padx=5)

speak("Voice Canva is ready")
root.mainloop()
