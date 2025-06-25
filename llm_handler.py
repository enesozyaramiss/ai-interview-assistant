from dotenv import load_dotenv
import os
import google.generativeai as genai

load_dotenv()

GEMINI_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_KEY)

generation_config = {
    "temperature": 0.8,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 750,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel("gemini-1.5-flash", generation_config=generation_config)

def get_llm_response(user_question):
    # HER çağrıda en güncel context.txt'yi oku
    with open("context.txt", "r", encoding="utf-8") as f:
        context_text = f.read()

    full_prompt = (
        f"Below is the user's CV:\n{context_text}\n\n"
        f"My question: {user_question}\n"
        f"Based on the CV, generate a professional, well-structured answer. "
        f"Do not write commentary or explanation, only the answer."
    )
    response = model.generate_content(full_prompt)
    return response.text.strip()
