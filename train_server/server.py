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

proxy_url = ''

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
    global proxy_url
    proxy_url = url
    return {"message": proxy_url}

@app.post('/ai/ocr', status_code=status.HTTP_200_OK)
async def ocr_image(file: UploadFile = File(...), token: str = Form()):
    if (token == "DKU18son") or (token=='Reader'):
        with tempfile.NamedTemporaryFile(delete=True) as temp_file:
            image_content = await file.read()
            temp_file.write(image_content)
            temp_file_path = temp_file.name
            ocr = OCR(
                file_dir=temp_file_path
            )
            ocr_result_, temp_file_path = rotate_image(ocr, temp_file_path)
            bbox_list = None
            word_list = None
            if ocr_result_:
                bbox_list, word_list = ocr_result_.get_data()
            else:
                bbox_list, word_list = ocr.get_data()

            temp_file.seek(0)
            image = Image.open(temp_file_path)
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
        return {'content': {'bbox': bbox_list, 'text': word_list, 'image': img_str}}
    else:
        return {
            "content": "fail",
            "message": "Please enter the correct token.",
        }

def send_training_request(user_id: str):
    response = requests.post('http://44.220.226.121:8080/generate', data={'user_id': user_id})
    response = requests.post(proxy_url + '/ai/train', data={'user_id': user_id})
    if response.status_code == 200:
        return {'content': response.json()['content']}
    else:
        return {'content': response.json()}

@app.post('/ai/train', status_code=status.HTTP_200_OK)
async def train(background_tasks: BackgroundTasks,user_id: str = Form()):
    background_tasks.add_task(send_training_request, user_id)
    return JSONResponse(content={'content': 'Training started'}, status_code=status.HTTP_200_OK)

@app.post('/ai/remain', status_code=status.HTTP_200_OK)
async def remain(user_id: str = Form()):
    response = requests.post(proxy_url + '/ai/remain', data={'user_id': user_id}).json()
    return JSONResponse(content={'content': response}, status_code=status.HTTP_200_OK)

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    error_message = {"error": str(exc)}
    return JSONResponse(status_code=500, content=error_message)
