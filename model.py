# app/model.py
from faster_whisper import WhisperModel
from TTS.api import TTS
from langchain.llms import Ollama

llm_model = Ollama(model="mistral:7b-instruct-q4_K_M")

# Use the smallest reasonable model for quality
# "tiny" = faster but less accurate
transcribe_model = WhisperModel("small", device="cpu", compute_type="int8")

speech_model = TTS(model_name="tts_models/en/ljspeech/glow-tts", progress_bar=False, gpu=False)