"""Local development entry point: python server.py."""

import logging

from backend.api.app import create_app

app = create_app()

if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host="127.0.0.1", port=8000)
