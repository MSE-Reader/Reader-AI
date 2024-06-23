from typing import List

import boto3
import pandas as pd
from fastapi import FastAPI, File, UploadFile, status, Form
from fastapi import BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.responses import FileResponse
from src.chat import Agent
from src.ocr import OCR, rotate_image
import requests
from PIL import Image
import base64
import io
import tempfile

app = FastAPI()

colab_url = ''

agent_dict = {}
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/set_url")
async def set_url(url:str = Form()):
    global colab_url
    colab_url = url
    return {"message": colab_url}

@app.post('/ai/remain', status_code=status.HTTP_200_OK)
async def remain(user_id: str = Form()):
    response = requests.post(colab_url + '/ai/remain', data={'user_id': user_id}).json()
    return JSONResponse(content={'content': response}, status_code=status.HTTP_200_OK)

@app.post('/ai/predict', status_code=status.HTTP_200_OK)
async def predict(user_id: str = Form()):
    response = requests.get(colab_url + '/ai/predict',data={'user_id':user_id})
    headers = {
        "Content-Disposition": 'attachment; filename="result.xlsx"'
    }
    return StreamingResponse(io.BytesIO(response.content), headers=headers, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

@app.post('/ai/chat', status_code=status.HTTP_200_OK)
async def chat(user_id: str = Form(), prompt:str = Form()):
    global agent_dict

    if user_id not in agent_dict:
        access_key_id = ""
        secret_access_key = ""
        bucket_name = ""
        s3_client = boto3.client("s3", aws_access_key_id=access_key_id, aws_secret_access_key=secret_access_key)
        file_path ='/home/ubuntu/server/result.xlsx'
        with open(file_path, 'wb') as f:
            s3_client.download_fileobj(bucket_name, (f'{user_id}/result/result.xlsx'),f)
        df = pd.read_excel(file_path)
        agent_dict[user_id] = Agent(df)

    image_path, response = agent_dict[user_id].run(prompt)
    img_str = False
    if image_path:
        image = Image.open(image_path)
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")  # 이미지 포맷에 따라 변경 가능
        img_str = base64.b64encode(buffered.getvalue()).decode()
    return {'image':img_str, 'response':response}


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    error_message = {"error": str(exc)}
    return JSONResponse(status_code=500, content=error_message)