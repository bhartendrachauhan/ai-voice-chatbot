from pydub import AudioSegment
import tempfile
from io import BytesIO
import soundfile as sf
from app.model import speech_model, transcribe_model, llm_model
import re
import webrtcvad
import wave
import numpy as np
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# Initialize a single in-memory chat history
history = InMemoryChatMessageHistory()

# Prompt template (unchanged)
response_template = PromptTemplate(
    input_variables=["text"],
    template="""
You are a general assistant.

Here's the user current query:
{text}

Generate a polite reply keeping the context of the conversation in mind.
Do not repeat yourself or reintroduce who you are.

Be straightforward.
""",
)

# Create the runnable sequence (replaces LLMChain)
chain = response_template | llm_model

# Wrap the chain with message history
chain_with_history = RunnableWithMessageHistory(
    chain,
    lambda: history,  # Single chat history instance
    input_messages_key="text",
    history_messages_key="chat_history",
)

async def generate_llm_response(text: str) -> str:
    # Invoke the chain asynchronously
    bot_response = await chain_with_history.ainvoke({"text": text})
    return bot_response


# 16kHz mono 16-bit PCM audio
vad = webrtcvad.Vad(3)  # Aggressiveness: 0 (very sensitive) to 3 (very aggressive)
buffer = BytesIO()


def pcm_to_wav(pcm_data: bytes, sample_rate=16000, sample_width=2, channels=1) -> BytesIO:
    buffer.seek(0)
    buffer.truncate(0)
    wav_io = buffer
    with wave.open(wav_io, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sample_width)  # 2 bytes = 16-bit PCM
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_data)
    wav_io.seek(0)
    return wav_io

# def is_silence(pcm_data: bytes, sample_rate=16000, threshold=0.5) -> bool:
    # frame_size = int(sample_rate * 0.02) * 2  # 20ms
    # speech_count = 0
    # total_frames = 0

    # for i in range(0, len(pcm_data), frame_size):
    #     frame = pcm_data[i:i + frame_size]
    #     if len(frame) < frame_size:
    #         break
    #     total_frames += 1
    #     if vad.is_speech(frame, sample_rate):
    #         speech_count += 1
            
    # if speech_count / total_frames > threshold:
    #     return False

    # return not vad.is_speech(pcm_data, sample_rate)

def is_silence(pcm_data: bytes, sample_rate=16000, frame_duration_ms=30, rms_thresh=500, zcr_thresh=0.05) -> bool:
    # Convert PCM bytes to int16 numpy array
    audio = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32)

    # --- RMS ---
    rms = np.sqrt(np.mean(audio ** 2))

    # --- ZCR ---
    signs = np.sign(audio)
    signs[signs == 0] = -1  # avoid zero-valued crossings
    zcr = np.mean(signs[:-1] != signs[1:])

    # --- Decision ---
    is_quiet = rms < rms_thresh
    is_low_zcr = zcr < zcr_thresh

    return is_quiet and is_low_zcr



def preprocess_audio(input_bytes: bytes) -> bytes:
    # Load audio from bytes
    audio = AudioSegment.from_file(BytesIO(input_bytes))

    # Convert to mono and set frame rate to 16kHz
    audio = audio.set_channels(1).set_frame_rate(16000)

    # Normalize to target dBFS
    target_dBFS = -20.0
    change_in_dBFS = target_dBFS - audio.dBFS
    audio = audio.apply_gain(change_in_dBFS)

    buffer.seek(0)
    buffer.truncate(0)
    audio.export(buffer, format="wav")
    buffer.seek(0)

    return buffer

def transcribe_text(input_bytes: bytes) -> str:
    # audio = AudioSegment.from_file(input_bytes)
    # buffer.seek(0)
    # buffer.truncate(0)
    # audio.export(buffer, format="wav")
    # buffer.seek(0)
    segments, _ = transcribe_model.transcribe(input_bytes, beam_size=1, language="en", vad_filter=True, vad_parameters={"threshold": 0.6})
    transcript = "".join([seg.text for seg in segments])
    return transcript
    

def generate_streamed_audio(text: str):
    sentences = split_text_by_fullstop(text)
    sample_rate = speech_model.synthesizer.output_sample_rate
    for sentence in sentences:
        # Generate audio for the sentence
        wav = speech_model.tts(sentence)
        buffer.seek(0)
        buffer.truncate(0)
        # Save to a BytesIO buffer
        sf.write(buffer, wav, sample_rate, format="WAV")
        buffer.seek(0)
        yield buffer.read()

def split_text_by_fullstop(text: str) -> list:
    # Split text by period followed by space or end of line
    sentences = re.split(r'(?<=[.?!])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]
    
    