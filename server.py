import app.patch_deps
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
import os
import signal

from app.bot import create_bot_pipeline
from app.transports.simple_websocket import SimpleWebsocketTransport

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    # Initialize Transport
    transport = SimpleWebsocketTransport(websocket=websocket)
    
    # Initialize Bot
    model_path = "models/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
    
    # Check if model exists
    if not os.path.exists(model_path):
        await websocket.send_text("Error: Model not found. Please run download_models.py")
        await websocket.close()
        return

    try:
        task = await create_bot_pipeline(transport, None, None, model_path)
    except Exception as e:
        print(f"Failed to create pipeline: {e}")
        # Try to notify client if connection is still open
        try:
            await websocket.send_text(f"Error initializing bot: {str(e)}")
            await websocket.close()
        except:
            pass
        return
    
    # Run pipeline
    try:
        await task.run()
    except Exception as e:
        print(f"Pipeline error: {e}")

# Serve static files
app.mount("/", StaticFiles(directory=".", html=True), name="static")

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
