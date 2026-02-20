
import os
import sys
import time
import platform
import datetime
import multiprocessing as mp
import threading
import queue
import subprocess
import psutil
import shutil
import warnings
import ctypes
GPIO_FAN = 17
GPIO_LIGHT = 27
import torch
import numpy as np
import joblib
import onnxruntime as ort
from transformers import (
    Wav2Vec2Processor,
    AutoModelForCTC,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    MarianMTModel,
    MarianTokenizer
)
from sklearn.pipeline import Pipeline
import stt
import vosk
from vosk import Model, KaldiRecognizer
import whisper
import pyaudio
import webrtcvad
import sounddevice
INTENT_MODEL_PATH = "models/intent_model_optimized.joblib"
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK = 320
VAD_MODE = 2
ASR_MODELS = {
    "coqui_stt": "models\ASR_STT\coqui_stt\coqui_stt_asr_model.tflite",
    "wav2vec2": "models\ASR_STT\IndicWav2Vec2\pytorch_model.bin",
    # "wav2vec2": "models\ASR_STT\wav2vec2\model.safetensors",
    "vosk": "models/ASR_STT/vosk-model-small-hi-0.22/vosk-model-small-hi-0.22",
    "whisper": "models\ASR_STT\Wisper-small\model.safetensors"
}
ASR_model_select="coqui_stt"
SYSTEM_TTS_ENGINES = {
    "espeak-ng": "models/espeak-ng",
    "festival": "models/festival"
}
TTS_model_select="espeak-ng"
TRANSLATION_MODEL={
    "hi-en_model": "models\Translation_model\hi-en_model"
}
TRANSLATION_model_select="hi-en_model"
def preload_asr_model():
    print("Loading ASR model...")
    asr_model_path = ASR_MODELS.get(ASR_model_select)
    
    if ASR_model_select == "coqui_stt":
        # Load Coqui STT (TensorFlow Lite)
        try:
            import stt
            session = stt.Model(asr_model_path)
            print("Coqui STT model loaded.")
        except Exception as e:
            print(f"Error loading Coqui STT model: {e}")
            return None
    elif ASR_model_select == "wav2vec2":
        # Load Wav2Vec2 (PyTorch-based)
        try:
            model = torch.load(asr_model_path)
            print("Wav2Vec2 model loaded.")
            return model
        except Exception as e:
            print(f"Error loading Wav2Vec2 model: {e}")
            return None
    elif ASR_model_select == "vosk":
        # Load Vosk (Kaldi-based)
        try:
            model = Model(asr_model_path)
            recognizer = KaldiRecognizer(model, SAMPLE_RATE)
            print("Vosk model loaded.")
            return recognizer
        except Exception as e:
            print(f"Error loading Vosk model: {e}")
            return None
    elif ASR_model_select == "whisper":
        # Load Whisper (OpenAI)
        try:
            import whisper
            model = whisper.load_model(asr_model_path)
            print("Whisper model loaded.")
            return model
        except Exception as e:
            print(f"Error loading Whisper model: {e}")
            return None
    return None
def preload_tts_model():
    print("Loading TTS model...")
    if TTS_model_select == "espeak-ng":
        try:
            # Check if espeak-ng is installed and available
            if shutil.which("espeak-ng") is not None:
                print("eSpeak-NG TTS model loaded.")
                return "espeak-ng"
            else:
                print("[MISSING] eSpeak-NG TTS engine not found.")
                return None
        except Exception as e:
            print(f"Error loading eSpeak-NG: {e}")
            return None
    elif TTS_model_select == "festival":
        try:
            # Check if Festival is installed and available
            if shutil.which("festival") is not None:
                print("Festival TTS model loaded.")
                return "festival"
            else:
                print("[MISSING] Festival TTS engine not found.")
                return None
        except Exception as e:
            print(f"Error loading Festival: {e}")
            return None
    return None
def preload_intent_model():
    print("Loading intent model...")
    model = joblib.load(INTENT_MODEL_PATH)
    print("Intent model loaded.\n")
    return model
def preload_translation_model():
    print("Loading translation model...")
    translation_model_path = TRANSLATION_MODEL.get(TRANSLATION_model_select)
    tokenizer = MarianTokenizer.from_pretrained(translation_model_path)
    translator = MarianMTModel.from_pretrained(translation_model_path)
    translator.eval()
    print("Hindi → English Translator model loaded. \n")
    return tokenizer,translator
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    asr_session = preload_asr_model()
    intent_model = preload_intent_model()
    tts_model = preload_tts_model()
    tokenizer,translator =preload_translation_model()
    print("All libraries imported successfully.")
    print("Models loaded.")
    print("System ready.")
WAKE_WORDS = [
    "hello assistant",
    "assistant",
    "सुनो",
    "नमस्ते असिस्टेंट",
    "hello",
    "namaste",
    "दोस्त", 
    "सुनो दोस्त",
    "नमस्ते",
    "नमस्ते कल्पना",
    "कल्पना",
    "kalpana"
]
SLEEP_WORDS = [
    "अलविदा दोस्त",
    "सो जाओ दोस्त",
    "ALVIDA",
    "BYE",
    "सो जाओ",
    "रुको",
    "stop",
    "go to sleep",
    "बंद करो"
]
IDLE = 0
ACTIVE = 1
class OfflineAssistant:
    def __init__(self, asr_session, intent_model):
        self.asr_session = asr_session
        self.intent_model = intent_model
        self.state = IDLE
        self.last_active_time = 0
        self.active_timeout = 20  # seconds before auto-sleep
        self.audio_queue = queue.Queue(maxsize=10)
        self.running = True
        self.lock = threading.Lock()
        print("Assistant initialized.")
    def check_wake_word(self, transcript):
        transcript = transcript.lower()
        for word in WAKE_WORDS:
            if word in transcript:
                return True
        return False
    def check_sleep_word(self, transcript):
        transcript = transcript.lower()
        for word in SLEEP_WORDS:
            if word in transcript:
                return True
        return False
    def activate(self):
        with self.lock:
            self.state = ACTIVE
            self.last_active_time = time.time()
            print("[STATE] ACTIVE")
    def deactivate(self):
        with self.lock:
            self.state = IDLE
            print("[STATE] IDLE")
    def check_timeout(self):
        if self.state == ACTIVE:
            if time.time() - self.last_active_time > self.active_timeout:
                print("Auto sleep triggered.")
                self.deactivate()
    def idle_listener(self):
        print("Idle listener started.")
        while self.running:
            try:
                audio_chunk = self.audio_queue.get(timeout=1)
                transcript = self.light_asr_inference(audio_chunk)
                if transcript and self.check_wake_word(transcript):
                    print("Wake word detected:", transcript)
                    self.activate()
            except queue.Empty:
                continue
    def active_listener(self):
        print("Active listener started.")
        while self.running:
            if self.state == ACTIVE:
                try:
                    audio_chunk = self.audio_queue.get(timeout=1)
                    transcript = self.full_asr_inference(audio_chunk)
                    if transcript:
                        print("Transcript:", transcript)
                        if self.check_sleep_word(transcript):
                            print("Sleep word detected.")
                            self.deactivate()
                            continue
                        self.last_active_time = time.time()
                        intent, conf = self.predict_intent(transcript)
                        print("Intent:", intent, "Confidence:", conf)
                        # self.execute_intent(intent)
                        self.handle_intent(intent, transcript)
                except queue.Empty:
                    pass
            self.check_timeout()
            time.sleep(0.05)
    def full_asr_inference(self, audio_chunk):
        input_data = np.array(audio_chunk, dtype=np.float32)
        inputs = {self.asr_session.get_inputs()[0].name: input_data}
        outputs = self.asr_session.run(None, inputs)
        transcript = self.decode_output(outputs)
        return transcript
    def decode_output(self, outputs):
        return ""
    def predict_intent(self, text):
        decision = self.intent_model.decision_function([text])
        confidence = np.max(decision)
        intent = self.intent_model.classes_[np.argmax(decision)]
        return intent, confidence
    def execute_intent(self, intent):
        print("Executing:", intent)
    def start(self):
        self.idle_thread = threading.Thread(target=self.idle_listener, daemon=True)
        self.active_thread = threading.Thread(target=self.active_listener, daemon=True)
        self.idle_thread.start()
        self.active_thread.start()
        print("Assistant threads started.")

        
def speak(self, text):
    if not text:
        return
def _speak_worker(msg):
    if TTS_model_select == "espeak-ng":
        subprocess.Popen(
            ["espeak-ng", "-v", "hi", msg],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
    elif TTS_model_select == "festival":
        subprocess.Popen(
            ["festival", "--tts"],
            stdin=subprocess.PIPE
        ).communicate(input=msg.encode())
threading.Thread(target=_speak_worker, args=(text,), daemon=True).start()
def get_time(self):
    now = datetime.datetime.now()
    response = f"अभी समय {now.hour} बजकर {now.minute} मिनट है"
    self.speak(response)
def get_date(self):
    today = datetime.datetime.now()
    response = f"आज तारीख {today.day}-{today.month}-{today.year} है"
    self.speak(response)
def check_weather(self):
    self.speak("मौसम सामान्य है")
def file_open(self):
    subprocess.Popen(["xdg-open", "."])
    self.speak("फाइल खोल दी गई है")
def file_close(self):
    subprocess.Popen(["pkill", "pcmanfm"])
    self.speak("फाइल बंद कर दी गई है")
def music_play(self):
    subprocess.Popen(["cvlc", "--random", "/home/pi/Music"])
    self.speak("संगीत चालू कर दिया गया है")
def music_next(self):
    subprocess.Popen(["pkill", "-SIGTERM", "vlc"])
    self.music_play()
def music_stop(self):
    subprocess.Popen(["pkill", "vlc"])
    self.speak("संगीत बंद कर दिया गया है")
def gpio_setup(self):
    import RPi.GPIO as GPIO
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(GPIO_FAN, GPIO.OUT)
    GPIO.setup(GPIO_LIGHT, GPIO.OUT)
    self.GPIO = GPIO
def fan_on(self):
    self.GPIO.output(GPIO_FAN, True)
    self.speak("पंखा चालू")
def fan_off(self):
    self.GPIO.output(GPIO_FAN, False)
    self.speak("पंखा बंद")
def light_on(self):
    self.GPIO.output(GPIO_LIGHT, True)
    self.speak("लाइट चालू")
def light_off(self):
    self.GPIO.output(GPIO_LIGHT, False)
    self.speak("लाइट बंद")
def translate_text(self, text):
    inputs = tokenizer(text, return_tensors="pt", padding=True)
    with torch.no_grad():
        output = translator.generate(**inputs, max_length=64)
    translated = tokenizer.decode(output[0], skip_special_tokens=True)
    self.speak(translated)
def web_search(self, query=""):
    subprocess.Popen(["chromium-browser", f"https://www.google.com/search?q={query}"])
    self.speak("ब्राउज़र खोल दिया गया है")
def set_timer(self, seconds=10):
    def timer_thread(sec):
        time.sleep(sec)
        self.speak("टाइमर पूरा हो गया")

    threading.Thread(target=timer_thread, args=(seconds,), daemon=True).start()
    self.speak("टाइमर शुरू")
def set_reminder(self, seconds=30, message="याद दिलाना है"):
    def reminder_thread(sec, msg):
        time.sleep(sec)
        self.speak(msg)
    threading.Thread(target=reminder_thread, args=(seconds, message), daemon=True).start()
    self.speak("रिमाइंडर सेट कर दिया गया है")
def tell_joke(self):
    jokes = [
        "टीचर: होमवर्क क्यों नहीं किया? छात्र: मैम, लाइट नहीं थी। टीचर: तो मोमबत्ती? छात्र: माचिस नहीं थी।",
        
        "डॉक्टर: आपको आराम की जरूरत है। ये नींद की गोलियाँ ले लीजिए। मरीज: डॉक्टर साहब, ये कब खानी हैं? डॉक्टर: अपनी नहीं, पत्नी की चाय में।",
        
        "पति: आज खाने में क्या है? पत्नी: जहर। पति: ठीक है, मैं देर से आऊंगा।",
        
        "बॉस: तुम्हें नौकरी से निकाला जाता है। कर्मचारी: पर क्यों? बॉस: क्योंकि तुम बहुत सवाल पूछते हो। कर्मचारी: कौन सा सवाल?",
        
        "दोस्त: भाई तू इतना पढ़ता क्यों है? दूसरा दोस्त: क्योंकि मैंने सुना है, मेहनत का फल मीठा होता है। और मुझे मीठा बहुत पसंद है।"
    ]
    self.speak(np.random.choice(jokes))
def introduce(self):
    self.speak(
        "मैं आपका ऑफलाइन हिंदी सहायक हूँ, पूरी तरह लोकल एआई पर आधारित। "
        "मेरा नाम कल्पना है। मैं RAC क्लब के 3 छात्रों द्वारा बनाया गया हूँ। "
        "मुझमें Python और Machine Learning के मॉडल्स इस्तेमाल किए गए हैं।"
    )
def calculate(self, text):
    if not text:
        self.speak("कृपया गणना बताएं")
        return
    text = text.lower()
    triggers = [
        "गणना करो", "कैल्कुलेट करो", "calculate", "ganana karo",
        "calculate karo", "calc", "ganana"
    ]
    for t in triggers:
        text = text.replace(t, "")
    text = text.strip()
    hindi_numbers = {
        "शून्य": 0, "एक": 1, "दो": 2, "तीन": 3, "चार": 4,
        "पांच": 5, "छह": 6, "सात": 7, "आठ": 8, "नौ": 9
    }
    english_numbers = {
        "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
        "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9
    }
    operators = {
        "plus": "+", "जोड़": "+", "प्लस": "+",
        "minus": "-", "घटाना": "-", "माइनस": "-",
        "multiply": "*", "multiplied": "*",
        "गुणा": "*", "गुना": "*",
        "divide": "/", "divided": "/",
        "भागा": "/", "भगा": "/"
    }
    tokens = text.split()
    num1 = None
    num2 = None
    operator = None
    for token in tokens:
        # Digit direct (e.g., 7)
        if token.isdigit():
            val = int(token)
            if val < 10:
                if num1 is None:
                    num1 = val
                else:
                    num2 = val
            continue
        # Hindi numbers
        if token in hindi_numbers:
            if num1 is None:
                num1 = hindi_numbers[token]
            else:
                num2 = hindi_numbers[token]
            continue
        # English numbers
        if token in english_numbers:
            if num1 is None:
                num1 = english_numbers[token]
            else:
                num2 = english_numbers[token]
            continue
        # Operators
        if token in operators:
            operator = operators[token]
    if num1 is None or num2 is None or operator is None:
        self.speak("सही गणना समझ नहीं पाया")
        return
    try:
        if operator == "+":
            result = num1 + num2
        elif operator == "-":
            result = num1 - num2
        elif operator == "*":
            result = num1 * num2
        elif operator == "/":
            if num2 == 0:
                self.speak("शून्य से भाग नहीं कर सकते")
                return
            result = round(num1 / num2, 2)
        else:
            self.speak("ऑपरेशन समझ नहीं पाया")
            return
        self.speak(f"परिणाम है {result}")
    except Exception:
        self.speak("गणना में त्रुटि हुई")
def handle_intent(self, intent, text=""):
    intent_map = {
        "time": self.get_time,
        "date": self.get_date,
        "weather": self.check_weather,
        "file_open": self.file_open,
        "file_close": self.file_close,
        "music_play": self.music_play,
        "music_next": self.music_next,
        "music_stop": self.music_stop,
        "gpio_fan_on": self.fan_on,
        "gpio_fan_off": self.fan_off,
        "gpio_light_on": self.light_on,
        "gpio_light_off": self.light_off,
        "translate_en": lambda: self.translate_text(text),
        "web_search": lambda: self.web_search(text),
        "timer": lambda: self.set_timer(10),
        "reminder": lambda: self.set_reminder(30),
        "tell_joke": self.tell_joke,
        "introduce": self.introduce,
        "calculate": lambda: self.calculate(text)
    }
    action = intent_map.get(intent)
    if action:
        action()
    else:
        self.speak("समझ नहीं पाया")
def startup_run(self):
    self.speak("सिस्टम शुरू हो गया है")
def run(self):
    self.startup_run()
    self.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        self.running = False
        print("Shutting down...")
