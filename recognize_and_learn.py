import json
import logging
import os
import pickle
import time
import webbrowser
from datetime import datetime
from io import BytesIO

import cv2
import face_recognition
import numpy as np
import pygame
import pyttsx3
import requests
import speech_recognition as sr
from PIL import Image

from training_faces import train_face_model

# === Setup Logging ===
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s')

# === Constants ===
DATASET_DIR = "facedata"
ENCODINGS_FILE = "encodings.pickle"
CONFIDENCE_THRESHOLD = 0.40
ACCURACY_TOLERANCE = 0.4
LOADING_AUDIO_PATH = "videoplayback.mp3"
THINKING_AUDIO_PATH = "thinking.mp3"
WELCOME_AUDIO_PATH = "welcome.mp3"
IP_CAM_URL = "http://192.168.151.51:8080/shot.jpg"

# === Globals ===
tts = pyttsx3.init()
pygame.mixer.init()
tokenizer = None
model = None
should_exit = False
is_playing = False
known_encodings = []
known_names = []
current_user = None
last_seen_user = None

SYSTEM_PROMPT = """You are Supervision, a friendly and clever AI assistant created by Bharath. You speak naturally 
like Jarvis from Iron Man. When a command is unclear, ask for clarification. Always be helpful and witty."""


# === Speech ===
def speak(text):
    logging.info(f"Assistant says: {text}")
    tts.say(text)
    tts.runAndWait()


def get_time_based_greeting():
    hour = datetime.now().hour
    if 5 <= hour < 12:
        return "Good morning"
    elif 12 <= hour < 17:
        return "Good afternoon"
    elif 17 <= hour < 21:
        return "Good evening"
    else:
        return "Hello"


# === Background Audio ===
def audio_playback(play=True, loop=True, audio_path="videoplayback.mp3"):
    global is_playing
    volume = 0.2
    if play and not is_playing:
        if os.path.exists(audio_path):
            pygame.mixer.music.load(audio_path)
            pygame.mixer.music.set_volume(volume)
            pygame.mixer.music.play(-1 if loop else 0)
            is_playing = True
        else:
            logging.warning(f"Audio file not found: {audio_path}")
    elif not play and is_playing:
        pygame.mixer.music.stop()
        is_playing = False


# === Model Loader ===
def load_api_model():
    audio_playback(True, loop=True)
    speak("Hello, I'm Supervision, your assistant developed by Bharath. Please wait while I warm up my brain.")

    max_retries = 10
    delay_seconds = 5
    ollama_url = "http://localhost:11434/api/tags"

    for i in range(max_retries):
        try:
            response = requests.get(ollama_url, timeout=3)
            if response.status_code == 200:
                audio_playback(False)
                speak("Model is ready. Let's identify who's here.")
                return
        except requests.exceptions.RequestException:
            logging.warning(f"Ollama not ready yet, retrying... ({i + 1}/{max_retries})")
            time.sleep(delay_seconds)

    audio_playback(False)
    speak("Sorry, I couldn't connect to the local brain. Please make sure the Ollama model is running.")
    logging.error("Ollama did not respond. Assistant will not work without it.")


# === Local LLM Interaction ===
def ask_local_llm(prompt):
    audio_playback(True, True, audio_path=THINKING_AUDIO_PATH)
    full_prompt = f"{SYSTEM_PROMPT}\nUser: {prompt}\nAI:"
    speak("Let me check that for you...")

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            headers={"Content-Type": "application/json"},
            data=json.dumps({
                "model": "llama3:instruct",
                "prompt": full_prompt,
                "stream": False,
                "num_predict": 150
            }),
            timeout=100
        )

        response.raise_for_status()
        result = response.json()
        answer = result.get("response", "").strip()
        logging.info(f"LLM response: {answer}")
        audio_playback(False)
        return answer

    except requests.exceptions.RequestException as e:
        audio_playback(False)
        logging.error(f"Ollama API error: {e}")
        return "Sorry, I couldn't get a response right now."


# === Command Listening ===
def listen_for_command(timeout=12):
    print("listening")
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        try:
            audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=timeout)
            command = recognizer.recognize_google(audio).lower()
            logging.info(f"Command received: {command}")
            return command
        except (sr.WaitTimeoutError, sr.UnknownValueError):
            return None
        except sr.RequestError:
            speak("Speech recognition service is unavailable.")
            return None


# === Process Command ===
def process_command(command):
    global should_exit

    if command is None:
        return

    if "exit" in command:
        speak("Shutting down. Goodbye.")
        should_exit = True
        exit()

    actions = {
        "time": lambda: speak(f"The time is {datetime.now().strftime('%I:%M %p')}"),
        "date": lambda: speak(f"Today is {datetime.now().strftime('%B %d, %Y')}"),
        "google": lambda: [webbrowser.open("https://www.google.com"), speak("Opening Google")],
        "youtube": lambda: [webbrowser.open("https://www.youtube.com"), speak("Opening YouTube")],
        "notepad": lambda: [os.system("start notepad.exe"), speak("Opening Notepad")],
        "calculator": lambda: [os.system("start calc.exe"), speak("Opening Calculator")],
    }

    for key in actions:
        if key in command:
            actions[key]()
            return

    try:
        if "open" in command:
            app_or_site = command.replace("open", "").strip()
            if "." in app_or_site:
                if not app_or_site.startswith("http"):
                    app_or_site = "http://" + app_or_site
                webbrowser.open(app_or_site)
                speak(f"Opening {app_or_site}")
            else:
                os.system(f"start {app_or_site}")
                speak(f"Trying to open {app_or_site}")
        else:
            reply = ask_local_llm(command)
            if reply == "Sorry, I couldn't get a response right now.":
                speak(reply)
                speak("Can I help you with any other thing.?")
                command = listen_for_command()
                process_command(command)
            else:
                speak(reply)

    except Exception as e:
        logging.error(f"Error processing command: {e}")
        speak("Sorry, something went wrong.")


# === Camera ===
def get_frame_from_ip_cam(ip_url=IP_CAM_URL):
    response = requests.get(ip_url)
    img = Image.open(BytesIO(response.content))
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


# === Encoding ===
def load_encodings():
    with open(ENCODINGS_FILE, "rb") as f:
        data = pickle.load(f)
    return data["encodings"], data["names"]


def retrain():
    train_face_model()


def update_encodings():
    global known_encodings, known_names
    known_encodings, known_names = load_encodings()


# === Face Recognition Loop ===
def face_recognition_loop():
    global known_encodings, known_names, current_user, last_seen_user
    greeting = get_time_based_greeting()
    load_api_model()

    while True:
        frame = get_frame_from_ip_cam()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(rgb)
        encodings = face_recognition.face_encodings(rgb, boxes)

        if not boxes and current_user:
            speak(f"{current_user} has left. Goodbye!")
            last_seen_user = current_user
            current_user = None

        user_found = False

        for (box, face_encoding) in zip(boxes, encodings):
            matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=ACCURACY_TOLERANCE)
            name = "Unknown"

            if True in matches:
                print("matches")
                matchedIdx = [i for i, b in enumerate(matches) if b]
                counts = {known_names[i]: known_names.count(known_names[i]) for i in matchedIdx}
                name = max(counts, key=counts.get)

                if name != current_user:
                    print("user found")
                    current_user = name
                    user_found = True

                    if name.lower() == "boss":
                        print("boss")
                        tts.setProperty("voice", tts.getProperty('voices')[1].id)
                        audio_playback(True, False, audio_path=WELCOME_AUDIO_PATH)
                        speak(f"{greeting}, Boss. Supervision is at your service. Great to see you back. Thanks for "
                              f"developing me!")
                        speak("How can I help you sir..?")
                        audio_playback(False)
                        print("done greeting")
                        break
                    else:
                        print("others")
                        tts.setProperty("voice", tts.getProperty('voices')[0].id)
                        speak(f"{greeting}, {name}. Welcome back! This is Supervision, your assistant developed by "
                              f"BHARATH.")
                        speak("How can I help you today..?")
                        print("done greeting")
                        break
                break

            elif name == "Unknown":
                tts.setProperty("voice", tts.getProperty('voices')[0].id)
                face_distances = face_recognition.face_distance(known_encodings, face_encoding)
                min_distance = min(face_distances) if len(face_distances) > 0 else 1.0

                if min_distance < CONFIDENCE_THRESHOLD and sum(matches) < 2:
                    speak("Hmm, you seem familiar, but I can't identify you yet.")
                    command = listen_for_command()
                    process_command(command)
                    continue

                speak("New face detected. Please tell me your name.")
                new_name = listen_for_command()

                if new_name:
                    speak(f"Did you say {new_name}? Say 'confirm' to confirm. Or 'cancel' to cancel")
                    confirm = listen_for_command()
                    if confirm == 'confirm':
                        new_dir = os.path.join(DATASET_DIR, new_name)
                        os.makedirs(new_dir, exist_ok=True)
                        for i in range(5):
                            speak(f"Capturing photo {i + 1}.")
                            img_path = os.path.join(new_dir, f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{i + 1}.jpg")
                            cv2.imwrite(img_path, frame)
                            cv2.waitKey(1000)
                        speak("Training system. One moment...")
                        audio_playback(True, True, audio_path=LOADING_AUDIO_PATH)
                        retrain()
                        update_encodings()
                        audio_playback(False)
                        speak("Restarting detection to include you.")
                        current_user = None
                        break
                    elif confirm == 'cancel':
                        speak("Registration Cancelled!")
                    else:
                        speak("Name not confirmed. Skipping registration.")

        # 🎯 Handle command loop outside face-detection for-loop
        if current_user and user_found:
            print("answering")
            while current_user:
                command = listen_for_command()
                process_command(command)

                if should_exit:
                    return

                # Check if user is still present
                frame = get_frame_from_ip_cam()
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                boxes = face_recognition.face_locations(rgb)
                encodings = face_recognition.face_encodings(rgb, boxes)

                match_found = False
                for enc in encodings:
                    matches = face_recognition.compare_faces(known_encodings, enc, tolerance=ACCURACY_TOLERANCE)
                    if True in matches and known_names[matches.index(True)] == current_user:
                        match_found = True
                        break

                if not match_found:
                    speak(f"{current_user} has left. Goodbye!")
                    last_seen_user = current_user
                    current_user = None
                    break

        # cv2.imshow("Face Recognition", frame)
        # if cv2.waitKey(1) & 0xFF == ord("q"):
        #     speak("Do you want to exit?")
        #     if listen_for_command() == "yes":
        #         break

    # cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        update_encodings()
        face_recognition_loop()
    except KeyboardInterrupt:
        speak("Assistant is shutting down. Goodbye!")
        exit()
