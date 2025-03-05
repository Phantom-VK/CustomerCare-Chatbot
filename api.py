import uvicorn
from model import VyomChatbot
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

chatbot = VyomChatbot()
# Create FastAPI app
app = FastAPI(
    title="Vyom Chatbot API",
    description="API for UBI Customer Care Chatbot",
    version="1.0.0"
)


# Request Model
class ChatRequest(BaseModel):
    query: str
    user_id: Optional[str] = None

# Response Model
class ChatResponse(BaseModel):
    response: str


# Chat Endpoint
@app.post("/chat", response_model=ChatResponse)
async def process_chat(request: ChatRequest):
    try:
        # Process the query
        response = chatbot.ask_question(request.query)

        # In a real-world scenario, you might want to calculate confidence
        # For now, we'll use a placeholder value

        return ChatResponse(
            response=response
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Error Handling Middleware
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return {
        "error": True,
        "status_code": exc.status_code,
        "message": exc.detail
    }

# Main entry point for running the server
if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )