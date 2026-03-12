



import subprocess
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class ChatRequest(BaseModel):
    model: str
    messages: list

@app.post("/v1/chat/completions")
async def chat_completion(request: ChatRequest):
    # Extracts the last user message
    prompt = request.messages[-1]["content"]
    
    # Run the official bitnet.cpp binary (adjust path if needed)
    # The 'run_inference.py' script is the standard way to invoke the binary
    cmd = [
        "python", "run_inference.py", 
        "-m", "models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf", 
        "-p", prompt, 
        "-n", "512"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    response_text = result.stdout.strip()

    return {
        "choices": [{
            "message": {"role": "assistant", "content": response_text}
        }]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
