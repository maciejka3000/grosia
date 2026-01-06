from fastapi import FastAPI, UploadFile, HTTPException, File
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from celery.result import AsyncResult
from tasks import task_crop_image
import os


app = FastAPI()
storage_path = 'storage'
os.makedirs(storage_path, exist_ok=True)
app.mount('/media', StaticFiles(directory=storage_path), name='media')

@app.post("/start/crop_image")
async def start_cropping(file: UploadFile = File(...)):
    content = await file.read()
    task = task_crop_image.delay(content)
    return {"task_id": task.id}