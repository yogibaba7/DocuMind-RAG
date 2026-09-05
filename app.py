"""
Main entry point for Streamlit application.
Loads and runs scripts/app.py.
"""
import os
import sys
import runpy

# Ensure workspace root and scripts are in sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.join(BASE_DIR, "scripts")

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

app_script = os.path.join(SCRIPTS_DIR, "app.py")

if __name__ == "__main__":
    runpy.run_path(app_script, run_name="__main__")
