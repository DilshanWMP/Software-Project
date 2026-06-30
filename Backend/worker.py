import os
import json
import time
import pika
import cv2
from sqlalchemy.orm import Session
from ultralytics import YOLO
import database

# Configuration
RABBITMQ_HOST = "localhost"
QUEUE_NAME = "video_processing_queue"
RESULTS_DIR = "C:\\Endoscopy\\Results"
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'CNN model', 'best.pt'))

os.makedirs(RESULTS_DIR, exist_ok=True)

# Load the model once
print(f"Loading YOLO model from {MODEL_PATH}...")
model = YOLO(MODEL_PATH)
print("Model loaded successfully.")

def process_media(task_id, file_path):
    print(f"Starting processing for task {task_id} at {file_path}")
    
    ext = os.path.splitext(file_path)[1].lower()
    is_image = ext in ['.jpg', '.jpeg', '.png']
    
    if is_image:
        frame = cv2.imread(file_path)
        if frame is None:
            raise Exception("Error opening image file")
            
        results = model(frame, verbose=False)
        annotated_frame = results[0].plot()
        
        # Extract detected classes
        detected = []
        for box in results[0].boxes:
            cls_id = int(box.cls[0].item())
            cls_name = results[0].names[cls_id]
            detected.append(cls_name)
        detected_str = ", ".join(list(set(detected))) if detected else "None"
        
        result_filename = f"res_{task_id}{ext}"
        result_path = os.path.join(RESULTS_DIR, result_filename)
        cv2.imwrite(result_path, annotated_frame)
        print(f"Finished processing image {task_id}. Saved to {result_path}")
        return result_path, detected_str
    
    # Video processing
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        raise Exception("Error opening video file")
        
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    result_filename = f"res_{task_id}.mp4"
    result_path = os.path.join(RESULTS_DIR, result_filename)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(result_path, fourcc, fps, (width, height))
    
    frame_count = 0
    all_detected = set()
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            results = model(frame, verbose=False)
            
            for box in results[0].boxes:
                cls_id = int(box.cls[0].item())
                all_detected.add(results[0].names[cls_id])
                
            annotated_frame = results[0].plot()
            out.write(annotated_frame)
            frame_count += 1
            
            if frame_count % 30 == 0:
                print(f"Processed {frame_count} frames for {task_id}")
                
    finally:
        cap.release()
        out.release()
        
    detected_str = ", ".join(list(all_detected)) if all_detected else "None"
    print(f"Finished processing video {task_id}. Saved to {result_path}")
    return result_path, detected_str

def callback(ch, method, properties, body):
    message = json.loads(body)
    video_id = message.get("videoId")
    file_path = message.get("filePath")
    
    print(f"Received message to process video: {video_id}")
    
    db = database.SessionLocal()
    task = db.query(database.VideoTask).filter(database.VideoTask.id == video_id).first()
    
    if not task:
        print(f"Task {video_id} not found in database. Skipping.")
        ch.basic_ack(delivery_tag=method.delivery_tag)
        db.close()
        return

    # Update status to processing
    task.status = "processing"
    db.commit()

    try:
        result_path, detected_str = process_media(video_id, file_path)
        
        # Update database with success
        task.status = "completed"
        task.result_path = result_path
        task.detected_classes = detected_str
        db.commit()
        print(f"Database updated to completed for {video_id} with classes: {detected_str}")
        
    except Exception as e:
        print(f"Error processing video {video_id}: {e}")
        task.status = "failed"
        db.commit()
        
    finally:
        db.close()
        ch.basic_ack(delivery_tag=method.delivery_tag)

def main():
    try:
        connection = pika.BlockingConnection(pika.ConnectionParameters(host=RABBITMQ_HOST))
        channel = connection.channel()
        channel.queue_declare(queue=QUEUE_NAME, durable=True)
        
        # Only send one message at a time to a worker
        channel.basic_qos(prefetch_count=1)
        
        channel.basic_consume(queue=QUEUE_NAME, on_message_callback=callback)
        
        print(' [*] Worker node is waiting for messages. To exit press CTRL+C')
        channel.start_consuming()
    except Exception as e:
        print(f"Failed to start RabbitMQ consumer: {e}")

if __name__ == '__main__':
    # Add a small delay on startup to wait for RabbitMQ if using docker-compose up --build
    time.sleep(2)
    main()
