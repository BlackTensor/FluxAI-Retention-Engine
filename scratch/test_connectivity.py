
import requests
import sys
import os

# Add the project root to sys.path to allow imports from utils
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.ollama_client import check_ollama

def test_connection():
    print("Checking AI Model File...")
    model_path = os.path.join('models', 'xgb_model.pkl')
    if os.path.exists(model_path):
        print(f"[OK] AI Engine Model Found: {model_path}")
    else:
        print(f"[FAIL] AI Engine Model NOT Found at {model_path}")

    print("\nChecking Ollama Connection...")
    is_running, models = check_ollama()
    if is_running:
        print("[OK] Ollama is Online")
        print(f"Available models: {', '.join(models)}")
    else:
        print("[FAIL] Ollama is Offline or not accessible at http://localhost:11434")

if __name__ == "__main__":
    test_connection()
