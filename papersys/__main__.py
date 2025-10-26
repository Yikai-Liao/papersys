"""Allow running papersys as a module: python -m papersys"""

from dotenv import load_dotenv
from papersys.cli import app
from papersys.const import BASE_DIR

if __name__ == "__main__":
    load_dotenv(BASE_DIR / ".env")
    app()
