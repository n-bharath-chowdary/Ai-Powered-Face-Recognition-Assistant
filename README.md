# 🤖 AI-Powered Face Recognition Assistant

![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

---

## 🚀 Project Overview
An advanced AI assistant system powered by real-time face recognition and voice interaction, designed to recognize individuals, communicate using speech, and adapt dynamically to new users. The system combines computer vision, natural language processing, and local LLMs to create a JARVIS-like experience.

---
💡 Why?
Manual authentication and robotic assistants are outdated. This project brings identity-aware AI to life with personalized voice interaction, auto-retraining, and local language model support, perfect for security, automation, and smart personal environments.

---
## 🎥 Demo Preview
|**Example**|
|---------------|
|"Hi Boss, welcome back. What can I do for you today?" (System auto-detects your face, greets you, and waits for your command...)|


Face Recognition Interface	Voice Assistant Interaction

---	

## 🔍 Features
✅ Real-time face recognition (using webcam or IP cam)

✅ Auto-retrain when a new face is detected

✅ Speech-to-text and text-to-speech communication

✅ Ollama LLM integration for dynamic, intelligent responses

✅ Personalized greetings based on recognized identity

✅ Efficient startup: model loads only once

✅ Retry mechanism if no voice command is detected

✅ Built-in control flow for multi-user interactions

---
## 🧠 Tech Stack
Python 3.10

OpenCV

face_recognition (Dlib)

pyttsx3 (Text-to-Speech)

speech_recognition

Ollama – Local LLM interface (JARVIS-style)

Streamlit (for optional GUI)

IP Webcam (optional)

---
## 🛠️ Installation
```
git clone https://github.com/n-bharath-chowdary/Ai-Powered-Face-Recognition-Assistant.git
cd Ai-Powered-Face-Recognition-Assistant
pip install -r requirements.txt
```
---
## Download the model:

[Click here to download model](https://drive.google.com/file/d/1De1VuqAXtGWdD4vW61J7NmUJFFvcgIja/view?usp=sharing)

Place encodings.pickle in the project root folder.

---
## ▶️ How to Run
Ensure your webcam or IP camera is connected and accessible.

Launch the app using:
```
python assistant.py
```

The assistant:

Detects and recognizes your face

Greets you and starts voice interaction

If the face is unknown, it prompts for a name and auto-trains on the new face

---
## 🧪 Algorithms Used
Face detection: HOG + CNN-based encoding

Real-time recognition: Euclidean distance on encodings

Voice interaction: Google Web Speech API + pyttsx3

LLM support: Ollama (local LLM API integration)

File management: Automatic folder creation and image storage

Dynamic learning: Self-retraining without restarting

---
### 📄 Project Documents
📘 [Final Report: Face_Recognition_Assistant_Report.pdf](https://github.com/n-bharath-chowdary/Ai-Powered-Face-Recognition-Assistant/blob/HIKE/documents/INTRODUCTION.pdf)

📊 [Presentation Slides: AI-Powered-Face-Assistant-Presentation.ppt](https://github.com/n-bharath-chowdary/Ai-Powered-Face-Recognition-Assistant/blob/HIKE/documents/AI-Based-Face-Recognition-Assistant.pptx)

📑 [Research Paper: Face_Recognition_Assistant_Research_Paper.pdf](https://github.com/n-bharath-chowdary/Ai-Powered-Face-Recognition-Assistant/blob/HIKE/documents/IJEDR2502050.pdf)

---
## ✨ Future Enhancements
🧠 Add emotion and sentiment analysis during interaction

🧾 Build logging system for attendance and visitor tracking

📱 Deploy on Raspberry Pi or Jetson Nano

🌐 Add cloud backup and remote monitoring support

🔐 Integrate with smart home security systems

---

## 🙋‍♂️ Author
#### Bharath Chowdary
##### [GitHub](https://github.com/n-bharath-chowdary) 
##### [LinkedIn](https://www.linkedin.com/in/n-bharath-chowdary/)
