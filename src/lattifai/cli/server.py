import os

import colorful
import uvicorn


def main():
    """Launch the LattifAI Web Interface."""
    print(colorful.bold_green("🚀 Launching LattifAI Web Interface..."))
    print(colorful.cyan("See http://localhost:8001"))

    # Ensure the directory contains the app
    # We might need to adjust python path or just rely on installed package

    uvicorn.run("lattifai.server.app:app", host="0.0.0.0", port=8001, reload=True, log_level="info")


if __name__ == "__main__":
    main()
