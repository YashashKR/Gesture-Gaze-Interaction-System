# rag_chatbot.py (MOCK/OFFLINE VERSION)
import random

def get_chatbot_response(user_query: str) -> str:
    """
    Simulates an AI response by matching keywords to the project knowledge base.
    Works 100% offline with zero latency.
    """
    query = user_query.lower().strip()

    # --- GREETINGS ---
    if any(word in query for word in ["hello", "hi", "hey", "start"]):
        return "👋 Hello! I am the Gesture Gaze Assistant. Ask me about:\n• Hand Gesture Mode\n• Eye Tracking Mode\n• Air Canvas\n• Snake Game"

    # --- HAND GESTURE MODE ---
    if "hand" in query or "gesture" in query:
        if "scroll" in query:
            return "📜 **Hand Scrolling:**\n• Scroll Down: Move Index finger *below* Middle finger.\n• Scroll Up: Move Index finger *above* Middle finger."
        if "click" in query or "pinch" in query:
            return "🖱️ **Hand Clicking:**\n• Left Click: Pinch your Thumb and Index finger together.\n• Drag: Hold the pinch and move your hand."
        return "🖐️ **Hand Mode Controls:**\n• Cursor: Follows your Index Finger tip.\n• Click: Pinch Thumb & Index.\n• Scroll: Move Index above/below Middle finger."

    # --- EYE TRACKING MODE ---
    if "eye" in query or "gaze" in query or "blink" in query or "wink" in query:
        if "scroll" in query or "wink" in query:
            return "📜 **Eye Scrolling:**\n• Scroll Down: Wink your **Left Eye**.\n• Scroll Up: Wink your **Right Eye**."
        if "click" in query or "blink" in query:
            return "🖱️ **Eye Clicking:**\n• Left Click: Perform a **Double Blink** (blink both eyes quickly)."
        return "👁️ **Eye Tracking Controls:**\n• Cursor: Follows your Iris center.\n• Click: Double Blink.\n• Scroll: Wink Left (Down) or Right (Up)."

    # --- AIR CANVAS (DRAWING) ---
    if "canvas" in query or "draw" in query or "paint" in query:
        if "hover" in query:
            return "✋ **Hover Mode:** Raise both Index and Middle fingers to move the cursor *without* drawing."
        if "color" in query or "palette" in query:
            return "🎨 **Colors:** Move your finger to the sidebar on the left to select different colors."
        if "tool" in query or "save" in query or "clear" in query:
            return "🛠️ **Canvas Tools:**\n• 'C': Clear Canvas\n• 'E': Eraser\n• 'U': Undo\n• 'S': Change Stroke Style\n• 'M': Toggle Dark Mode"
        return "🎨 **Air Canvas:**\n• Draw: Raise *only* your Index Finger.\n• Hover: Raise Index + Middle fingers.\n• Shortcuts: 'C' to Clear, 'S' to Save."

    # --- SNAKE GAME ---
    if "game" in query or "snake" in query:
        return "🐍 **Snake Game Controls:**\n• Point UP 👆: Move Up\n• Point DOWN 👇: Move Down\n• Point LEFT 👈: Move Left\n• Point RIGHT 👉: Move Right"

    # --- TECH STACK & SYSTEM ---
    if "tech" in query or "stack" in query or "code" in query:
        return "💻 **Tech Stack:**\n• Language: Python (Flask)\n• Vision: OpenCV & MediaPipe\n• GUI: HTML/CSS (Glassmorphism)\n• Game: Pygame"
    
    if "limitation" in query or "problem" in query:
        return "⚠️ **System Limitations:**\n• Requires good lighting.\n• Eye tracking needs a steady head position.\n• Performance depends on webcam quality."

    # --- FALLBACK RESPONSE ---
    return "🤖 I can help with Hand Mode, Eye Mode, Air Canvas, or the Game. Try asking: 'How do I click?' or 'How to draw?'"