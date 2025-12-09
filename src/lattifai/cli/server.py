import argparse
import os

import colorful
import uvicorn


def main():
    """Launch the LattifAI Web Interface."""
    parser = argparse.ArgumentParser(description="LattifAI Backend Server")
    parser.add_argument(
        "-p",
        "--port",
        type=int,
        default=8001,
        help="Port to run the server on (default: 8001)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind the server to (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--no-reload",
        action="store_true",
        help="Disable auto-reload on code changes",
    )

    args = parser.parse_args()

    print(colorful.bold_green("🚀 Launching LattifAI Backend Server..."))
    print(colorful.cyan(f"Server running at http://localhost:{args.port}"))
    print(colorful.yellow(f"Host: {args.host}"))
    print(colorful.yellow(f"Auto-reload: {'disabled' if args.no_reload else 'enabled'}"))
    print()

    uvicorn.run("lattifai.server.app:app", host=args.host, port=args.port, reload=not args.no_reload, log_level="info")


if __name__ == "__main__":
    main()
