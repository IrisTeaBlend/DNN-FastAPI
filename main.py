from fastapi import FastAPI, Form, Request, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.responses import StreamingResponse
from pathlib import Path
import os
import wave
import uuid  # 一意のファイル名を生成するために利用
from ttslearn.dnntts import DNNTTS

app = FastAPI()

# テンプレートと静的ファイルの設定
BASE_DIR = Path(__file__).resolve().parent
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
    if not text:
        raise HTTPException(status_code=400, detail="テキストが提供されていません")

    try:
        # 音声合成を実行
        wav, sr = engine.tts(text)

        # 一意のファイル名を生成
        unique_filename = f"{uuid.uuid4()}.wav"
        output_path = upload_dir / unique_filename

        # 音声データを保存
        with wave.open(str(output_path), 'wb') as wf:  # Pathオブジェクトをstrに変換
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(wav.tobytes())

        # 保存した音声ファイルをレスポンスとして返却
        return StreamingResponse(open(output_path, "rb"), media_type="audio/wav")

    except Exception as e:
        # 例外をキャッチしてエラー詳細を返す
        raise HTTPException(status_code=500, detail=f"音声合成中にエラーが発生しました: {e}")
