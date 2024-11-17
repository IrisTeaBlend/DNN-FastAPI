from fastapi import FastAPI, Form, Request, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.responses import StreamingResponse
from pathlib import Path
import os
import wave
from ttslearn.dnntts import DNNTTS

app = FastAPI()

# テンプレートと静的ファイルの設定
BASE_DIR = Path(__file__).resolve().parent  # 現在のファイルのディレクトリ
templates = Jinja2Templates(directory=BASE_DIR / "templates")

# モデルディレクトリの相対パス
model_dir = BASE_DIR / "tts_models/jsut_sr16000_duration_dnn_acoustic_dnn_sr16k"
engine = DNNTTS(model_dir)

# ファイルを保存するディレクトリ
upload_dir = BASE_DIR / "uploads"
os.makedirs(upload_dir, exist_ok=True)

@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/synthesize/")
async def synthesize_text(request: Request, text: str = Form(...)):
    if text:
        wav, sr = engine.tts(text)

        # 音声ファイルを一時的に保存
        output_path = upload_dir / "output.wav"
        with wave.open(output_path, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(wav.tobytes())

        # 保存した音声ファイルをストリーム化してレスポンスとして返す
        return StreamingResponse(open(output_path, "rb"), media_type="audio/wav")

    raise HTTPException(status_code=400, detail="テキストが提供されていません")
