"""Validate source checksums and the numerical/method unit tests from any cwd."""
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
if __name__=='__main__':
    subprocess.run([sys.executable,str(ROOT/'code/main.py'),'--verify-only'],check=True)
    subprocess.run([sys.executable,'-m','unittest','discover','-s',str(ROOT/'tests'),'-v'],check=True)
