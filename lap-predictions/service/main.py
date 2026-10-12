from fastapi import FastAPI
app = FastAPI(title="Laparoscopy Predictions")

@app.get("/health")
def health():
    return {"status": "ok"}