import os
import uuid
import json
import pika
from fastapi import FastAPI, UploadFile, File, Depends, HTTPException
from sqlalchemy.orm import Session
import database

app = FastAPI(title="Endoscopy AI API")

UPLOAD_DIR = "C:\\Endoscopy\\Uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# RabbitMQ configuration
RABBITMQ_HOST = "localhost"
QUEUE_NAME = "video_processing_queue"

def get_rabbitmq_channel():
    connection = pika.BlockingConnection(pika.ConnectionParameters(host=RABBITMQ_HOST))
    channel = connection.channel()
    channel.queue_declare(queue=QUEUE_NAME, durable=True)
    return connection, channel

@app.post("/api/videos/upload")
async def upload_video(file: UploadFile = File(...), db: Session = Depends(database.get_db)):
    if not file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.jpg', '.jpeg', '.png')):
        raise HTTPException(status_code=400, detail="Invalid file type")

    video_id = str(uuid.uuid4())
    file_extension = os.path.splitext(file.filename)[1]
    saved_filename = f"vid_{video_id}{file_extension}"
    file_path = os.path.join(UPLOAD_DIR, saved_filename)

    # Save file to disk
    with open(file_path, "wb") as buffer:
        content = await file.read()
        buffer.write(content)

    # Create database entry
    new_task = database.VideoTask(
        id=video_id,
        filename=file.filename,
        filepath=file_path,
        status="pending_ai_scan"
    )
    db.add(new_task)
    db.commit()

    # Publish to RabbitMQ
    try:
        connection, channel = get_rabbitmq_channel()
        message = {
            "videoId": video_id,
            "filePath": file_path,
            "status": "pending_ai_scan"
        }
        channel.basic_publish(
            exchange='',
            routing_key=QUEUE_NAME,
            body=json.dumps(message),
            properties=pika.BasicProperties(
                delivery_mode=2,  # make message persistent
            ))
        connection.close()
    except Exception as e:
        print(f"Failed to publish to RabbitMQ: {e}")
        new_task.status = "failed"
        db.commit()
        raise HTTPException(status_code=500, detail="Failed to queue video for processing")

    return {"message": "Video uploaded, processing in background.", "videoId": video_id}

@app.get("/api/videos/{video_id}/status")
def get_status(video_id: str, db: Session = Depends(database.get_db)):
    task = db.query(database.VideoTask).filter(database.VideoTask.id == video_id).first()
    if not task:
        raise HTTPException(status_code=404, detail="Video not found")
    
    return {
        "videoId": task.id,
        "status": task.status,
        "resultPath": task.result_path,
        "detectedClasses": task.detected_classes
    }
