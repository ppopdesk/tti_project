from pathlib import Path

#Output and Cache directory for medical dataset
OUTPUT_DIR = Path("data/medqa_en")
CACHE_DIR = Path("data/hf_cache")

#Ollama endpoint and model
OLLAMA_URL = "http://localhost:11434"
MODEL = "llama3.2:latest"

#Test Data file
TEST_FILE = "data/medqa_en/test.jsonl"