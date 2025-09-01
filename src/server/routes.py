from fastapi import FastAPI
from src.controller import agent_route
from src.server.server import app
def initialize_routes(app: FastAPI):
    """Function to initialize all the endpoint routes using included routers."""
    app.include_router(agent_route, prefix="/api")
    @app.get("/health")
    def health_check():
        return {"status": "ok"}
