from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import torch
import logging
import soundfile as sf
import io
import base64
import uvicorn
import random
import logging
from server.hifi_gan import infer_e2e, setup_models
from server.warmup_data import get_warmup_sentences

logging.basicConfig(level=logging.INFO)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize models and move to device
    setup_models()
    logging.info("Models loaded and ready for inference")
    
    try:
        warmup_texts = get_warmup_sentences(20)
        
        # Run inference with warmup texts
        logging.info("Performing warm-up inference...")
        await text_to_speech(TTSRequest(texts=warmup_texts, return_format="base64"))
        logging.info("Warm-up complete")
        
    except Exception as e:
        logging.error(f"Warm-up failed: {str(e)}")
    
    yield  # Server is now running and handling requests

app = FastAPI(title="TTS API Service", lifespan=lifespan)

class TTSRequest(BaseModel):
    texts: List[str]
    return_format: str = "base64"  # "base64" or "raw"

class TTSResponse(BaseModel):
    audio_data: List[str]  # Base64 encoded audio or raw waveform data
    sample_rate: int = 22050
    duration_seconds: List[float]

@app.post("/tts", response_model=TTSResponse)
async def text_to_speech(request: TTSRequest):
    try:
        # Run inference
        raw_audio, pre_pad_len, real_len, durs_predicted = infer_e2e(request.texts)
        
        audio_data = []
        durations = []
        
        # Process each audio in batch
        for i in range(len(request.texts)):
            pred_noise_duration = (real_len[i] - durs_predicted[i, pre_pad_len[i].long():].sum()) * 256
            audio = raw_audio.cpu().detach().numpy()[i, :int(pred_noise_duration)]
            
            # Convert to WAV bytes
            buffer = io.BytesIO()
            sf.write(buffer, audio, 22050, format='WAV')
            buffer.seek(0)
            
            audio_b64 = base64.b64encode(buffer.read()).decode()
            audio_data.append(audio_b64)
            durations.append(len(audio) / 22050)  # Duration in seconds
            
        return TTSResponse(
            audio_data=audio_data,
            duration_seconds=durations
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
