# app/main.py
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import StreamingResponse
import os
from app.utils import preprocess_audio, generate_streamed_audio, is_silence, transcribe_text, pcm_to_wav, generate_llm_response
from io import BytesIO
import time

app = FastAPI()



@app.post("/transcribe", response_model=dict)
async def transcribe_binary_audio(request: Request):
    try:
        audio_bytes = await request.body()
        transcript = preprocess_audio_and_generate_transcript(audio_bytes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to transcribe: {str(e)}")

    return {"transcript": transcript}

@app.post("/speak")
async def speak(request: Request):
    try:
        data = await request.json()
        text = data.get("text", "")

        if not text:
            return {"error": "No text provided"}
        return StreamingResponse(generate_streamed_audio(text), media_type="audio/wav")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to transcribe: {str(e)}")

@app.websocket("/ws/audio")
async def audio_ws(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket connected")

    audio_buffer = b""
    silence_buffer = b""
    transcript = ""
    # last_transcription_time = time.time()
    CHUNK_SIZE_8_SEC = 16000*2*8
    CHUNK_SIZE_4_SEC = 16000*2*1
    CHUNK_SIZE_1_MIN = 16000*2*60
    # silence_start = None
    # silence_duration = 3

    try:
        while True:
            frame = await websocket.receive_bytes()
            if frame:
                # now = time.time()
                audio_buffer += frame
                silence_buffer += frame

                # # Interrupt long speech (40s)
                # if now - start_time > 40:
                #     await websocket.send_text("Speech too long. Interrupting.")
                #     break
                
                # Silence detection
                if len(silence_buffer) >= CHUNK_SIZE_4_SEC:
                    wav_io_last_4_sec = pcm_to_wav(silence_buffer)
                    last_4_sec_transcription = transcribe_text(wav_io_last_4_sec).strip()
                    if not last_4_sec_transcription:
                        # Final chunk (leftover audio)
                        if audio_buffer:
                            last_wav_io = pcm_to_wav(audio_buffer)
                            text = transcribe_text(last_wav_io)
                            transcript += f" {text}"
                            audio_buffer = b""

                        # await websocket.send_text(f"final: {transcript.strip()}")
                        transcript = transcript.strip()
                        if transcript:
                            bot_response = await generate_llm_response(transcript)
                            # await websocket.send_text(f"Bot response: {bot_response.strip()}")
                            for chunk in generate_streamed_audio(bot_response):
                                await websocket.send_bytes(chunk)
                            
                            transcript = ""
                            audio_buffer = b""
                    silence_buffer = b""
                # if is_silence(frame):
                #     if silence_start == None:
                #         silence_start = now
                #     elif now - silence_start >= silence_duration:
                #         # Final chunk (leftover audio)
                #         if audio_buffer:
                #             last_wav_io = pcm_to_wav(audio_buffer)
                #             text = transcribe_text(last_wav_io)
                #             transcript += f" {text}"
                #             audio_buffer = b""

                #         # await websocket.send_text(f"final: {transcript.strip()}")
                #         if transcript:
                #             bot_response = generate_llm_response(transcript)
                #             # await websocket.send_text(f"Bot response: {bot_response.strip()}")
                #             for chunk in generate_streamed_audio(bot_response):
                #                 await websocket.send_bytes(chunk)
                        
                #             transcript = ""
                #             audio_buffer = b""
                #             # silence_buffer = b""
                # else:
                #     silence_start = None
                
                if len(audio_buffer) >= CHUNK_SIZE_8_SEC:
                    chunk = audio_buffer[:CHUNK_SIZE_8_SEC]
                    audio_buffer = audio_buffer[CHUNK_SIZE_8_SEC:]
                    wav_io = pcm_to_wav(chunk)
                    text = transcribe_text(wav_io)
                    transcript += f" {text}"
            # Progressive transcription every 5s
            # preprocessed_audio_stream = preprocess_audio(audio_buffer)
            # preprocessed_audio_buffer = preprocessed_audio_stream.getvalue()
            # if len(preprocessed_audio_buffer) >= CHUNK_SIZE:
            #     chunk = bytes(preprocessed_audio_buffer[:CHUNK_SIZE])  # Convert to bytes if needed
            #     del audio_buffer[:CHUNK_SIZE]
            #     text = transcribe_text(preprocessed_audio_stream)
            #     transcript += f" {text}"
            #     await websocket.send_text(f"partial: {text}")

        # Send transcript to Mistral (coming next)
        # response_text = await query_mistral(transcript.strip())

    except Exception as e:
        print(e)
        await websocket.send_text(f"Error: {e}")
    finally:
        await websocket.close()
        print("WebSocket closed")
    