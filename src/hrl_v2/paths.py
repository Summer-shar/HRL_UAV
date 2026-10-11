"""Installed code/resources are read-only; workspaces hold run outputs."""
from pathlib import Path

def source_root():
    return Path(__file__).resolve().parent.parent

def default_config():
    return Path(__file__).resolve().parent/"configs/v2_3_0.json"
