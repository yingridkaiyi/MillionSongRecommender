import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from src.webapp.app import create_app

app = create_app()

if __name__ == '__main__':
    app.run(debug=True, port=5000)