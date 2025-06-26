import streamlit as st
import sounddevice as sd
import numpy as np
from utils import transcribe_audio, record_audio
from llm_handler import get_llm_response
from PyPDF2 import PdfReader
from scipy.signal import resample

# Küresel değişkenler
recording_data = []
recording = False
samplerate = 48000

def auto_select_loopback_device():
    devices = sd.query_devices()
    for i, d in enumerate(devices):
        if d['max_input_channels'] > 0:
            name = d['name'].lower()
            if any(k in name for k in ["stereo", "mix", "loopback", "missaggio", "what u hear"]):
                return i, d['name']
    return None, None

def render_ui():
    global recording_data, recording
    st.set_page_config(page_title="🎤 Interview AI Assistant", layout="centered")
    apply_custom_styles()

    st.markdown("<h1 style='text-align: center;'>🤖 <span style='color:#4A90E2'>Interview AI Assistant</span></h1>", unsafe_allow_html=True)

    device_id, device_name = auto_select_loopback_device()

    st.markdown("### 🎙 Record an audio question")
    if st.button("🔴 Start Recording"):
        recording_data = []
        recording = True
        st.info("🔴 Recording started... Click '⏹ Stop Recording' when done.")
        start_recording(device_id)

    if st.button("⏹ Stop Recording"):
        if recording:
            recording = False
            st.info("🛑 Recording stopped. Processing audio...")
            stop_and_process_recording()


    st.markdown("### 📄 Upload your CV (PDF or TXT)")
    cv_file = st.file_uploader("", type=["pdf", "txt"])
    if cv_file:
        process_cv(cv_file)


def start_recording(device_id):
    def callback(indata, frames, time_info, status):
        if recording:
            recording_data.append(indata.copy())
    st.session_state['stream'] = sd.InputStream(samplerate=samplerate, channels=1, device=device_id, callback=callback)
    st.session_state['stream'].start()

def stop_and_process_recording():
    stream = st.session_state.get('stream')
    if stream:
        stream.stop()
        stream.close()
    if not recording_data:
        st.warning("⚠ No audio captured.")
        return

    combined = np.concatenate(recording_data, axis=0).flatten()
    target_samples = int(len(combined) * 16000 / samplerate)
    resampled = resample(combined, target_samples).astype(np.float32)

    transcription = transcribe_audio(resampled)
    st.markdown(f"**📝 Transcription:** {transcription}")
    if transcription and len(transcription) >= 1:
        with st.spinner("💡 Generating AI response..."):
            answer = get_llm_response(transcription)
            st.markdown(f"**🤖 AI Response:** {answer}")
    else:
        st.warning("⚠ Transcription too short or empty.")


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
        #st.text_area("📄 Extracted CV (preview)", cv_text[:1000], height=300)
    else:
        st.warning("⚠ CV could not be processed or was empty.")


def apply_custom_styles():
    st.markdown("""
        <style>
            .stButton > button {
                background-color: #4A90E2;
                color: white;
                padding: 0.75em 1.5em;
                border-radius: 10px;
                font-size: 1.1em;
                transition: background-color 0.3s ease;
            }
            .stButton > button:hover {
                background-color: #357ABD;
            }
        </style>
    """, unsafe_allow_html=True)


def handle_audio_flow(device_id, device_name, duration):
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
