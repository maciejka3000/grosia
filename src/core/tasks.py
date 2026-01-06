import time
from celery import Celery
import io
from PIL import Image
from sympy import im
import uuid
import os

storage_path = 'storage'
celery_app = Celery('worker', 
                    broker='amqp://guest:guest@localhost:5672//', 
                    backend='rpc://')

@celery_app.task
def task_crop_image(file_bytes: bytes):
    print('detecting receipt, cropping image...')
    
    image = Image.open(io.BytesIO(file_bytes))
    # processing goes there
    time.sleep(3)
    
    image = image.crop((100, 100, 400, 400))
    filename = f"cropped_{uuid.uuid4()}.jpg"
    file_path = os.path.abspath(os.path.join(storage_path, filename))
    image.save(file_path)
    
    return {"status": "done", "image_path": file_path}


