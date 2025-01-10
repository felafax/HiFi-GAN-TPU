from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import torch
import logging
import soundfile as sf
import io
import base64
import uvicorn

from hifi_gan import infer_e2e, setup_models

app = FastAPI(title="TTS API Service")

class TTSRequest(BaseModel):
    texts: List[str]
    return_format: str = "raw"  # "base64" or "raw"

class TTSResponse(BaseModel):
    audio_data: List[str]  # Base64 encoded audio or raw waveform data
    sample_rate: int = 22050
    duration_seconds: List[float]

@app.on_event("startup")
async def startup_event():
    # Initialize models and move to device
    setup_models()
    logging.info("Models loaded and ready for inference")

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
            
            if request.return_format == "base64":
                # Convert to base64
                audio_b64 = base64.b64encode(buffer.read()).decode()
                audio_data.append(audio_b64)
            else:
                # Return raw waveform as list
                audio_data.append(audio.tolist())
                
            durations.append(len(audio) / 22050)  # Duration in seconds
            
        return TTSResponse(
            audio_data=audio_data,
            duration_seconds=durations
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
