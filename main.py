import os
import requests
import subprocess
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Using Hugging Face's serverless API to keep the hosting free and fast
API_URL = "https://api-inference.huggingface.co/models/Rikorose/deepfilternet"
HF_TOKEN = os.getenv("HF_TOKEN") # We will set this in Render settings

@app.post("/enhance")
async def enhance_audio(file: UploadFile = File(...)):
    raw_data = await file.read()
    
    # 1. AI Denoising
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    response = requests.post(API_URL, headers=headers, data=raw_data)
    
    temp_clean = "clean.wav"
    with open(temp_clean, "wb") as f:
        f.write(response.content)

    # 2. Studio Vocal Chain (FFmpeg)
    # This chain adds: High-pass (80Hz), EQ Air boost (12kHz), and Soft Compression
    output_path = "studio_vocal.wav"
    ffmpeg_cmd = (
        f"ffmpeg -y -i {temp_clean} "
        f"-af 'highpass=f=80, equalizer=f=12000:t=q:w=1:g=3, "
        f"compand=attacks=0:points=-80/-80|-20/-12|0/-10' {output_path}"
    )
    subprocess.run(ffmpeg_cmd, shell=True)
    
    return FileResponse(output_path, media_type="audio/wav")

@app.get("/")
def read_root():
    return FileResponse("index.html")
