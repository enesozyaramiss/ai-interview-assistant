import sounddevice as sd
import numpy as np
from scipy.signal import resample
import whisper

model = whisper.load_model("small")

def record_audio(duration, samplerate=48000, device=None):
    stt_data = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, device=device)
    sd.wait()
    audio = stt_data.flatten()

    if audio.size == 0 or np.all(audio == 0):
        return None

    target_samples = int(len(audio) * 16000 / samplerate)
    resampled = resample(audio, target_samples).astype(np.float32)
    return resampled

def transcribe_audio(audio):
    result = model.transcribe(audio)
    return result.get("text", "").strip()
