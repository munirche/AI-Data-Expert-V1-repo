"""Launch the AI Data Entry V1 Streamlit app."""

import subprocess
import sys

subprocess.run([
    sys.executable, "-m", "streamlit", "run",
    "Code/data_entry/app_data_entry_v1.py",
])
