"""
Load environment variables from .env file.
Call load_env() at the start of any script that needs DATABASE_URL or other config.
"""

import os
from pathlib import Path


def load_env():
    """Load environment variables from .env file in the project root."""
    env_file = Path(__file__).parent / ".env"
    
    if not env_file.exists():
        print(f"Warning: .env file not found at {env_file}")
        print("Create one by copying .env.example: cp .env.example .env")
        return
    
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            # Skip comments and empty lines
            if not line or line.startswith("#"):
                continue
            # Parse KEY=VALUE
            if "=" in line:
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip()
                # Set environment variable
                os.environ[key] = value


if __name__ == "__main__":
    load_env()
    print(f"DATABASE_URL: {os.environ.get('DATABASE_URL', 'NOT SET')}")
