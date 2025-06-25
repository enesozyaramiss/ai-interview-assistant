import streamlit as st
import sounddevice as sd
import threading
from overlay_gui import create_overlay_app
from utils import record_audio, transcribe_audio
from llm_handler import get_llm_response
from PyPDF2 import PdfReader


def auto_select_loopback_device():
    devices = sd.query_devices()
    for i, d in enumerate(devices):
        if d['max_input_channels'] > 0:
            name = d['name'].lower()
            if any(k in name for k in ["stereo", "mix", "loopback", "missaggio", "what u hear"]):
                return i, d['name']
    return None, None

def render_ui():
    st.set_page_config(page_title="Interview AI Assistant", layout="centered")
    st.title("🎤 Interview AI Assistant")

    device_id, device_name = auto_select_loopback_device()

    if device_id is not None:
        st.success(f"✅ Auto-selected device: {device_name}")
    else:
        st.error("❌ No suitable loopback/stereo mix device found!")

    # Overlay GUI Başlat
    if st.button("🖥 Start Overlay GUI"):
        if device_id is not None:
            threading.Thread(target=create_overlay_app, args=(device_id,), daemon=True).start()
            st.success(f"Overlay GUI started with device: {device_name}")
        else:
            st.error("❌ Cannot start overlay without valid input device.")

    # CV Yükleme
    cv_file = st.file_uploader("📄 Upload your CV (PDF or TXT)", type=["pdf", "txt"])
    if cv_file:
        process_cv(cv_file)


def process_cv(cv_file):
    cv_text = ""
    if cv_file.type == "application/pdf":
        reader = PdfReader(cv_file)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                cv_text += text + "\n"
    else:
        cv_text = cv_file.read().decode("utf-8")

    if cv_text.strip():
        with open("context.txt", "w", encoding="utf-8") as f:
            f.write(cv_text)
        st.success("✅ CV processed and context updated.")
        st.text_area("📄 Extracted CV (preview)", cv_text[:1000], height=300)
    else:
        st.warning("⚠ CV could not be processed or was empty.")

def handle_audio_flow(device_id, device_name, duration):
    if device_id is not None:
        st.info(f"Recording from: {device_name}")
        audio = record_audio(duration, device=device_id)
        if audio is not None:
            transcription = transcribe_audio(audio)
            st.markdown(f"**📝 Transcription:** {transcription}")
            if transcription and len(transcription) >= 3:
                with st.spinner("💡 Generating AI response..."):
                    answer = get_llm_response(transcription)
                    st.markdown(f"**🤖 AI Response:** {answer}")
            else:
                st.warning("⚠ Transcription too short or empty.")
        else:
            st.warning("⚠ No audio captured.")
    else:
        st.error("❌ No valid input device found.")

def handle_text_question(user_q):
    if user_q.strip():
        with st.spinner("💡 Generating AI response..."):
            answer = get_llm_response(user_q)
            st.markdown(f"**🤖 AI Response:** {answer}")
    else:
        st.warning("⚠ Please enter a question.")
