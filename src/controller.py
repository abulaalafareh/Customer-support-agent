from src.server.server import app
from starlette.concurrency import run_in_threadpool
import os
from fastapi import Request, HTTPException, APIRouter
from src.agents import Agent

agent_route = APIRouter(prefix="/agents", tags=["agents"])


@agent_route.get("/health")
async def health():
    print("here")
    # Basic sanity check for API key (optional)
    has_key = bool(os.getenv("OPENAI_API_KEY"))
    return {"status": "ok", "openai_key_set": has_key}

@agent_route.post("/tool_agent")
async def tool_agent_endpoint(request: Request):
    try:
        data = await request.json()
        # Agent.query_tool_agent is a class-level function (no self), so call directly
        result = await run_in_threadpool(Agent.query_tool_agent, data["query"])
        return {"answer": result}
    except Exception as e:
        # Surface the error for quick debugging
        raise HTTPException(status_code=500, detail=str(e))

