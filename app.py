from flask import Flask, render_template, redirect, url_for, request, jsonify
import hand_gesture_mouse
import eye_tracking_mouse
import rag_chatbot
import drawing_canvas   
#import eye_tracking_canvas


app = Flask(__name__)
mode = "hand"

# ----------------- UI ROUTES -----------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/start/<tracking_mode>")
def start_detection(tracking_mode):
    global mode
    mode = tracking_mode  # Set mode (hand/eye/canvas)

    if mode == "hand":
        hand_gesture_mouse.running = True
        hand_gesture_mouse.detect_hand_gestures()
    elif mode == "eye":
        eye_tracking_mouse.running = True
        eye_tracking_mouse.detect_eye_tracking()
    elif mode == "canvas": 
        drawing_canvas.start_canvas()

    return redirect(url_for("index"))

@app.route("/stop")
def stop_detection():
    hand_gesture_mouse.running = False
    eye_tracking_mouse.running = False
    drawing_canvas.stop_canvas()   
    return redirect(url_for("index"))

# ----------------- CHATBOT ROUTE -----------------
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_query = data.get("message", "")  
    if not user_query:
        return jsonify({"response": "I didn’t get your question."})

    bot_response = rag_chatbot.get_chatbot_response(user_query)
    return jsonify({"response": bot_response})

# ----------------- RUN -----------------
if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
