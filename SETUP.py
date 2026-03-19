import os

# Ensure utf-8 output to avoid charmap encode errors on Windows
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')


# Definition der Ordnerstruktur
folders = [
    "data/raw",
    "data/processed",
    "data/labels",
    "models",
    "notebooks",
    "src",
]

# Dateien mit Initialinhalt
files = {
    "requirements.txt": "torch\ntorchvision\nopencv-python\npandas\nmatplotlib\nscikit-learn",
    "README.md": "# Game Aim Localization Project\nPhase 1: Environment & Data Collection.",
    "src/dataset.py": "# Woche 4: PyTorch Dataset Loader",
    "src/model.py": "# Woche 5: ResNet18 Architecture",
    "src/train.py": "# Woche 6: Training Loop",
    "inference_demo.py": "# Woche 8: Live Demo Script"
}

def setup():
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
        print(f"Ordner erstellt: {folder}")
    
    for file_path, content in files.items():
        with open(file_path, "w") as f:
            f.write(content)
        print(f"Datei erstellt: {file_path}")

    print("\nFertig! Dein Projekt ist bereit für Woche 1.")

if __name__ == "__main__":
    setup()

print("done")