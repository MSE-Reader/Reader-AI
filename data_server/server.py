import json
from typing import List

from fastapi import FastAPI, status, Form, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from src import DataGenerator
from PIL import Image, ImageFile
import boto3
ImageFile.LOAD_TRUNCATED_IMAGES = True

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://100.25.242.208:8080/"],
    allow_credentials=True,
    allow_methods=["*"],  
)
import os
import shutil

def delete_folder_and_contents(folder_path):
    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        shutil.rmtree(folder_path)
        print(f"Deleted folder and all contents: {folder_path}")
    else:
        print(f"Folder does not exist: {folder_path}")


@app.post('/upload', status_code=status.HTTP_200_OK)
async def ocr_image(user_id: str = Form(),files: List[UploadFile] = File(...)):
    access_key_id = ""
    secret_access_key = ""
    bucket_name = ""

    s3_client = boto3.client("s3", aws_access_key_id=access_key_id, aws_secret_access_key=secret_access_key)
    for file in files:
        file_content = await file.read()
        s3_client.put_object(Bucket=bucket_name, Key=f'{user_id}/{file.filename}', Body=file_content)
    return {'content':'True'}

@app.post('/generate', status_code=status.HTTP_200_OK)
async def ocr_image(user_id: str = Form()):
    access_key_id = ""
    secret_access_key = ""
    bucket_name = ""

    s3_client = boto3.client("s3", aws_access_key_id=access_key_id, aws_secret_access_key=secret_access_key)
    obj_list = s3_client.list_objects(Bucket=bucket_name, Prefix=user_id)
    ocr_result = []

    original_directory = '/home/ubuntu/server/temp'
    delete_folder_and_contents(original_directory)
    os.makedirs(original_directory, exist_ok=False)
    for file_info in obj_list['Contents']:
        if file_info['Key'].split('.')[-1] in ['jpg','PNG','png','JPG','jpeg','JPEG','webp']:
            print(file_info['Key'])
            file_name = file_info['Key'].split('/')[-1].split('.')[0] + '.jpg'
            response = s3_client.get_object(Bucket=bucket_name, Key=(file_info['Key']))
            response = response['Body']
            Image.open(response).convert('RGB').save(os.path.join(original_directory,file_name), 'JPEG')
        elif '.json' in file_info['Key']:
            response = s3_client.get_object(Bucket=bucket_name, Key=(file_info['Key']))
            response = json.loads(response['Body'].read())
            file_name = file_info['Key'].split('/')[-1].split('.')[0] + '.jpg'
            response['file_name'] = file_name
            ocr_result.append(response)

    del s3_client
    access_key_id = ""
    secret_access_key = ""
    bucket_name = ""
    s3_client = boto3.client("s3", aws_access_key_id=access_key_id, aws_secret_access_key=secret_access_key)
    background_dir = './src/background'

    DataGenerator(ocr_result, original_directory, background_dir, s3_client,bucket_name,user_id, count=200).run()
    return {'content': 'True'}

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    error_message = {"error": str(exc)}
    return JSONResponse(status_code=500, content=error_message)