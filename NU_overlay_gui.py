import tkinter as tk
import threading
import numpy as np
import sounddevice as sd
from scipy.signal import resample
import whisper
import keyboard
import time
import queue
from llm_handler import get_llm_response

model_whisper = whisper.load_model("small")
samplerate = 48000
device = None
recording = False
recording_data = []
msg_queue = queue.Queue()

def create_overlay_app(selected_device):
    global device
    device = selected_device
    root, text_widget = create_overlay()
    text_widget.tag_config("header", foreground="blue", font=("Consolas", 14, "bold"))

    # Başlat queue process döngüsü
    root.after(100, lambda: process_queue(root, text_widget))

    # Key listener thread başlat
    threading.Thread(target=key_listener, daemon=True).start()

    root.mainloop()

def create_overlay():
    root = tk.Tk()
    root.title("Real-time Q&A Overlay")
    root.attributes('-alpha', 0.95)
    root.attributes('-topmost', True)
    root.geometry("700x400+100+100")

    text_widget = tk.Text(
        root,
        font=("Consolas", 16, "bold"),
        bg="#F0F0F0",
        fg="black",
        wrap='word',
        padx=10,
        pady=10
    )
    text_widget.pack(expand=True, fill='both')
    text_widget.insert('end', "System ready. Press F9 to start recording, K to stop.\n\n")
    text_widget.config(state='disabled')

    return root, text_widget

def process_queue(root, text_widget):
    while not msg_queue.empty():
        header, message = msg_queue.get()
        text_widget.config(state='normal')
        text_widget.insert('end', "\n" + "=" * 60 + "\n")
        if header:
            text_widget.insert('end', f"{header}\n", "header")
        text_widget.insert('end', message + "\n")
        text_widget.see('end')
        text_widget.config(state='disabled')
    root.after(100, lambda: process_queue(root, text_widget))

def send_to_queue(message, header=None):
    msg_queue.put((header, message))

def start_recording():
    global recording, recording_data
    recording = True
    recording_data = []
    send_to_queue("Recording started...", "INFO")
    print("Recording started...")

    def callback(indata, frames, time_info, status):
        if recording:
            recording_data.append(indata.copy())

    stream = sd.InputStream(samplerate=samplerate, channels=2, device=device, callback=callback)
    stream.start()
    return stream

def stop_recording(stream):
    global recording
    recording = False
    stream.stop()
    stream.close()

    if not recording_data:
        send_to_queue("No audio captured.", "INFO")
        print("No audio captured.")
        return

    combined = np.concatenate(recording_data, axis=0)
    combined = combined[:, 0].astype(np.float32)
    target_samples = int(len(combined) * 16000 / samplerate)
    resampled = resample(combined, target_samples)

    result = model_whisper.transcribe(resampled, language="en")
    text = result["text"].strip()

    if text:
        send_to_queue(text, "Transcription")
        print(f"Transcription: {text}")

        send_to_queue("Getting AI response...", "INFO")
        try:
            answer = get_llm_response(text)
            send_to_queue(answer, "AI Response")
            print(f"AI Response: {answer}")
        except Exception as e:
            send_to_queue(f"LLM error: {e}", "ERROR")
            print(f"LLM error: {e}")
    else:
        send_to_queue("No speech detected or transcription empty.", "INFO")
        print("No speech detected or transcription empty.")

def key_listener():
    stream = None
    while True:
        if keyboard.is_pressed('esc'):
            send_to_queue("Exiting.", "INFO")
            print("Exiting.")
            if stream:
                stream.stop()
                stream.close()
            break

        if keyboard.is_pressed('f9') and not recording:
            stream = start_recording()
            while keyboard.is_pressed('f9'):
                time.sleep(0.2)

        if keyboard.is_pressed('k') and recording:
            stop_recording(stream)
            while keyboard.is_pressed('k'):
                time.sleep(0.2)

        time.sleep(0.05)
