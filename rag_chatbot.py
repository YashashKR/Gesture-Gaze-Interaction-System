# rag_chatbot.py
import google.generativeai as genai
from config import GEMINI_API_KEY

genai.configure(api_key=GEMINI_API_KEY)

with open("knowledge_base/hand_eye_mouse_info.txt", "r", encoding="utf-8") as f:
    knowledge_base = f.read()

def get_chatbot_response(user_query: str) -> str:
    prompt = f"""
You are a helpful assistant for the Multimodal Air Canvas project.
Answer questions using the following knowledge base:

{knowledge_base}

Rules for answering:
- Reply in **2–3 short sentences max**.
- Reply in **bullet points** (•).
- Be clear, simple, and to the point.
- Keep it **short and clear** (1–2 lines per point).
- If the question is unrelated, politely say: "I can only answer about Air Canvas, Gesture Mode, or Eye Tracking."

User: {user_query}
Assistant:
"""
    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(prompt)

    # ✅ Handle possible response formats
    if hasattr(response, "text") and response.text:
        return response.text.strip()
    elif hasattr(response, "candidates") and response.candidates:
        try:
            return response.candidates[0].content.parts[0].text.strip()
        except:
            pass
    return "⚠️ Sorry, I couldn’t generate a response."
