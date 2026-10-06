"""Convenient root entry point; equivalent to python code/main.py --analyze."""
from pathlib import Path
import runpy
import sys

if __name__ == '__main__':
    code_dir = Path(__file__).resolve().parent / 'code'
    sys.path.insert(0, str(code_dir))
    runpy.run_path(str(code_dir / 'main.py'), run_name='__main__')
