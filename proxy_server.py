from fastapi import FastAPI, File, UploadFile, status, Form, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
from typing import List
import boto3
import pickle
import shutil
import asyncio
import pandas as pd
from io import BytesIO
import ngrok
import os
import ast

executor = None
ml_models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global executor
    # Load the ML model
    print('Model Loading . . .')
    model_directory = '/content/drive/MyDrive/캡스톤/reciept_workspace_reciept/Work_Space/Model/model'
    model = PredictModel(model_directory)
    ml_models["model"] = model
    executor = ThreadPoolExecutor(max_workers=6)
    yield
    # Clean up the ML models and release the resources
    print('SHUT DOWN')
    executor.shutdown(wait=True)
    ml_models.clear()

app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

def delete_folder_and_contents(folder_path):
    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        shutil.rmtree(folder_path)
        print(f"Deleted folder and all contents: {folder_path}")
    else:
        print(f"Folder does not exist: {folder_path}")

def train_model(user_id):
    original_directory = f'./{user_id}'
    delete_folder_and_contents(original_directory)
    os.makedirs(original_directory, exist_ok=False)

    log_path = os.path.join(original_directory, 'log.txt')
    with open(log_path, 'w') as f:
            f.write(f"{{ 'percentage':'0%', 'remain' :'loading . . .'}}")

    access_key_id = ""
    secret_access_key = ""
    bucket_name = ""

    s3_client = boto3.client("s3", aws_access_key_id=access_key_id, aws_secret_access_key=secret_access_key)
    obj_list = s3_client.list_objects(Bucket=bucket_name, Prefix=user_id)

    data_config = None
    for file_info in obj_list['Contents']:
        if '.jpg' in file_info['Key']:
            print(file_info['Key'])
            response = s3_client.get_object(Bucket=bucket_name, Key=(file_info['Key']))
            with open(os.path.join(original_directory, file_info['Key'].split('/')[-1]), 'wb') as f:
              s3_client.download_fileobj(bucket_name, (file_info['Key']), f)
        elif '.pkl' in file_info['Key']:
            response = s3_client.get_object(Bucket=bucket_name, Key=(file_info['Key']))
            data_config = pickle.loads(response['Body'].read())
    train_dataset, eval_dataset, label_list, processor = PrepareDataset(data_config, original_directory).prepare_data()

    del s3_client
    access_key_id = ""
    secret_access_key = ""
    bucket_name = ""
    s3_client = boto3.client("s3", aws_access_key_id=access_key_id, aws_secret_access_key=secret_access_key)
    module = TrainModel(processor, train_dataset, eval_dataset, label_list, s3_client, bucket_name, user_id)
    module.train(original_directory, log_path)

@app.post('/ai/train', status_code=status.HTTP_200_OK)
async def train(background_tasks: BackgroundTasks,user_id: str = Form()):
    background_tasks.add_task(train_model, user_id)
    return {'status': True}

@app.post('/ai/remain', status_code=status.HTTP_200_OK)
async def train(user_id: str = Form()):
    access_key_id = ""
    secret_access_key = ""
    bucket_name = ""

    original_directory = f'./{user_id}'
    s3_client = boto3.client("s3", aws_access_key_id=access_key_id, aws_secret_access_key=secret_access_key)
    obj_list = s3_client.list_objects(Bucket=bucket_name, Prefix=user_id+'/model')
    if 'Contents' in obj_list:
      return { 'percentage':'학습이 완료되었습니다.', 'remain' :'학습이 완료되었습니다.'}
    if not os.path.exists(original_directory):
      return { 'percentage':'모델을 학습해주세요.', 'remain' :'모델을 학습해주세요.'}
    log_path = os.path.join(original_directory, 'log.txt')
    with open(log_path, 'r') as f:
      log = f.read()
    return ast.literal_eval(log)

@app.get('/ai/predict', status_code=status.HTTP_200_OK)
async def predict(user_id: str = Form()):

    access_key_id = ""
    secret_access_key = ""
    bucket_name = ""

    original_directory = '/content/temp'

    s3_client = boto3.client("s3", aws_access_key_id=access_key_id, aws_secret_access_key=secret_access_key)
    obj_list = s3_client.list_objects(Bucket=bucket_name, Prefix=user_id+'/model')
    model_directory = os.path.join(original_directory, 'model')
    delete_folder_and_contents(model_directory)
    os.makedirs(model_directory, exist_ok=False)
    for file_info in obj_list['Contents']:
      with open(os.path.join(model_directory, file_info['Key'].split('/')[-1]), 'wb') as f:
              s3_client.download_fileobj(bucket_name, (file_info['Key']), f)

    ml_models["model"] = PredictModel(model_directory)
    obj_list = s3_client.list_objects(Bucket=bucket_name, Prefix=user_id+'/predict')

    image_directory = os.path.join(original_directory, 'image')
    delete_folder_and_contents(image_directory)
    os.makedirs(image_directory, exist_ok=False)
    for file_info in obj_list['Contents']:
        if '.jpg' in file_info['Key']:
            print(file_info['Key'])
            with open(os.path.join(image_directory, file_info['Key'].split('/')[-1]), "wb") as f:
              s3_client.download_fileobj(bucket_name, (file_info['Key']), f)
            s3_client.delete_object(Bucket =bucket_name,Key=file_info['Key'])


    files = os.listdir(image_directory)
    tasks = []
    for file_name in files:
        file_path = os.path.join(image_directory, file_name)
        task = process_single_image(file_path)
        tasks.append(task)

    ocr_list = await asyncio.gather(*tasks)
    result = ml_models["model"].predict(ocr_list)

    file_path = os.path.join(image_directory, "result.xlsx")
    df = pd.DataFrame(result)
    df.to_excel(file_path, index=False)

    with open(file_path, 'rb') as f:
        s3_client.put_object(Bucket=bucket_name, Key=(f'{user_id}/result/result.xlsx'), Body=f)
    headers = {'Content-Disposition': f'attachment; filename="result.xlsx"'}
    return FileResponse(file_path, headers=headers)