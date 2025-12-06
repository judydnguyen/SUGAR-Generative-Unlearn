"""
Configuration file for SUGAR Generative Unlearning project.
"""
import os
from pathlib import Path

# Get project root directory (parent of this config file)
GLOBAL_PATH = os.getenv(
    "SUGAR_PROJECT_ROOT",
    str(Path(__file__).parent.absolute())
)