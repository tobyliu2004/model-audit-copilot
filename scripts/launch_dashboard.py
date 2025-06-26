"""Launch the Model Audit Copilot dashboard."""

import os
import sys
import subprocess
import argparse


def main():
    """Launch the Streamlit dashboard with proper configuration."""
    parser = argparse.ArgumentParser(description="Launch Model Audit Copilot Dashboard")
    parser.add_argument("--port", type=int, default=8501, help="Port to run dashboard on")
    parser.add_argument("--host", default="localhost", help="Host to bind to")
    parser.add_argument("--config", help="Path to configuration file")
    
    args = parser.parse_args()
    
    # Set environment variables if config provided
    if args.config:
        os.environ["AUDIT_CONFIG_PATH"] = args.config
    
    # Get the dashboard path
    dashboard_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "dashboard",
        "main_app.py"
    )
    
    # Launch streamlit
    cmd = [
        sys.executable, "-m", "streamlit", "run",
        dashboard_path,
        "--server.port", str(args.port),
        "--server.address", args.host
    ]
    
    print(f"Launching Model Audit Copilot Dashboard on http://{args.host}:{args.port}")
    
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nDashboard stopped.")


if __name__ == "__main__":
    main()