"""Allow running the CLI as a module: python -m papersys.cli"""

from . import app
from ..const import BASE_DIR
from dotenv import load_dotenv

if __name__ == "__main__":
    load_dotenv(BASE_DIR / ".env")
    app()
