#!/usr/bin/env python3
import os

def create_init_files(root_dir):
    """
    Rekurzívan létrehozza az __init__.py fájlokat minden mappában,
    ha azok még nem léteznek.
    """
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if '__init__.py' not in filenames:
            init_file = os.path.join(dirpath, '__init__.py')
            try:
                with open(init_file, 'w') as f:
                    f.write("# Ez a fájl biztosítja, hogy ez a könyvtár csomagként legyen felismerve.\n")
                print(f"Létrehozva: {init_file}")
            except Exception as e:
                print(f"Hiba a {init_file} létrehozása közben: {e}")

if __name__ == "__main__":
    project_root = os.path.abspath(os.path.dirname(__file__))
    create_init_files(project_root)
