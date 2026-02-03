"""Launches Lance Data Viewer in Docker to inspect LanceDB databases."""

import argparse
import glob
import os
import subprocess
import sys


PROJECT_PATH = "C:/Users/munir/Projects/AI Data Expert V1"
IMAGE = "ghcr.io/gordonmurray/lance-data-viewer:lancedb-0.24.3"
PORT = 8080
CONTAINER_NAME = "lance-viewer"

DATABASES = {
    "v1": "expert_learning_system_v1_db",
    "v2": "expert_learning_system_v2_db"
}


def stop_existing():
    """Stop any container using our port or name."""
    subprocess.run(
        ["docker", "rm", "-f", CONTAINER_NAME],
        capture_output=True
    )

    result = subprocess.run(
        ["docker", "ps", "--filter", f"publish={PORT}", "-q"],
        capture_output=True,
        text=True
    )

    container_ids = result.stdout.strip().split()
    for cid in container_ids:
        if cid:
            subprocess.run(["docker", "rm", "-f", cid], capture_output=True)


def find_lance_tables():
    """Find all .lance tables in all databases."""
    tables = []
    for db_key, db_folder in DATABASES.items():
        db_path = os.path.join(PROJECT_PATH, db_folder)
        if os.path.exists(db_path):
            # Find .lance folders inside the database
            lance_dirs = glob.glob(os.path.join(db_path, "*.lance"))
            for lance_dir in lance_dirs:
                table_name = os.path.basename(lance_dir)
                # Create unique name: v1_tablename.lance or v2_tablename.lance
                display_name = f"{db_key}_{table_name}"
                tables.append({
                    "path": lance_dir.replace("\\", "/"),
                    "display_name": display_name
                })
    return tables


def list_databases():
    """List available databases and tables."""
    print("Available databases:")
    for key, path in DATABASES.items():
        full_path = os.path.join(PROJECT_PATH, path)
        exists = "exists" if os.path.exists(full_path) else "not found"
        print(f"  {key}: {path} ({exists})")

    print("\nLance tables found:")
    tables = find_lance_tables()
    for t in tables:
        print(f"  {t['display_name']}")


def main():
    parser = argparse.ArgumentParser(description="Launch Lance Data Viewer")
    parser.add_argument("db", nargs="?", default="all", choices=["v1", "v2", "all"],
                        help="Database to view: v1, v2, or all (default: all)")
    parser.add_argument("--list", action="store_true", help="List available databases")
    args = parser.parse_args()

    if args.list:
        list_databases()
        return

    print("Stopping any existing viewer...")
    stop_existing()

    if args.db == "all":
        # Mount all lance tables with unique names
        tables = find_lance_tables()

        if not tables:
            print("No lance tables found.")
            return

        print(f"\nStarting Lance Data Viewer...")
        print(f"Tables: {', '.join(t['display_name'] for t in tables)}")
        print(f"Open in browser: http://localhost:{PORT}")
        print("Press Ctrl+C to stop.\n")

        cmd = [
            "docker", "run", "--rm",
            "--name", CONTAINER_NAME,
            "-p", f"{PORT}:8080"
        ]

        # Add volume mount for each table
        for t in tables:
            cmd.extend(["-v", f"{t['path']}:/data/{t['display_name']}:ro"])

        cmd.append(IMAGE)
    else:
        db_folder = DATABASES[args.db]
        db_path = f"{PROJECT_PATH}/{db_folder}"

        if not os.path.exists(db_path):
            print(f"Error: Database not found at {db_path}")
            list_databases()
            return

        print(f"\nStarting Lance Data Viewer...")
        print(f"Database: {db_folder}")
        print(f"Open in browser: http://localhost:{PORT}")
        print("Press Ctrl+C to stop.\n")

        cmd = [
            "docker", "run", "--rm",
            "--name", CONTAINER_NAME,
            "-p", f"{PORT}:8080",
            "-v", f"{db_path}:/data:ro",
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
