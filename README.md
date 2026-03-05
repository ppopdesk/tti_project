# Confidence-Based Classifier

## Setup Instructions

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

### 1. Prepare MedQA Dataset

Run the data preparation script to download and process the MedQA dataset:

```bash
python prepare_medqa_en_split.py
```

This will:
- Download the MedQA dataset from HuggingFace
- Split it into train/validation/test sets
- Save processed data to `data/medqa_en/`

**Note:** If you need to change the output directory or cache directory, modify the `OUTPUT_DIR` and `CACHE_DIR` variables in `config.py`.

### 2. Setup Ollama and Download Model

1. Download and install [Ollama](https://ollama.ai)
2. Pull the required model (e.g., llama3.2):

```bash
ollama pull llama3.2:latest
```

3. Start the Ollama server:

```bash
ollama serve
```

**Note:** If you need to change the model name or Ollama endpoint, update the `MODEL` and `OLLAMA_URL` variables in `config.py`.

### 3. Run the Experiment

Run the main experiment script which implements the "think on disagreement" algorithm:

```bash
python llama_exp.py
```

This will process the test dataset and output results showing model performance and confidence-based analysis.

## Project Structure

- `prepare_medqa_en_split.py` - Dataset preparation and download
- `llama_exp.py` - Main experiment script with confidence-based classifier
- `requirements.txt` - Python dependencies
- `config.py` - Contains some variables
