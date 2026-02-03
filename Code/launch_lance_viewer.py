"""Launches Lance Data Viewer in Docker to inspect the LanceDB database."""

import subprocess
import sys


DB_PATH = "C:/Users/munir/Projects/AI Data Expert V1/expert_learning_system_v1_db"
IMAGE = "ghcr.io/gordonmurray/lance-data-viewer:lancedb-0.24.3"
PORT = 8080


def main():
    print(f"Starting Lance Data Viewer...")
    print(f"Database: {DB_PATH}")
    print(f"Open in browser: http://localhost:{PORT}")
    print("Press Ctrl+C to stop.\n")

    cmd = [
        "docker", "run", "--rm",
        "-p", f"{PORT}:{PORT}",
        "-v", f"{DB_PATH}:/data:ro",
        IMAGE
    ]

    try:
        subprocess.run(cmd)
    except FileNotFoundError:
        print("Error: Docker not found. Make sure Docker Desktop is running.")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
