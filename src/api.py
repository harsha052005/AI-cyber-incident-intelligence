from fastapi import FastAPI

app = FastAPI(title="AI Cyber Incident Intelligence Platform")


@app.get("/")
def home():
    return {
        "message": "AI Cyber Incident Intelligence Platform is running!",
        "status": "success"
    }
