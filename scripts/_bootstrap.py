import os
import sys


def bootstrap() -> str:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    for path in (project_root, script_dir):
        if path not in sys.path:
            sys.path.insert(0, path)

    return project_root
