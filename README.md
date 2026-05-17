# Test-Time Inference for Medical Datasets

Scaling LLMs to improve reasoning in specialized domains such as medicine is computationally expensive and often impractical. Test-time compute and domain-specific fine-tuning offer a more efficient path to improved performance; however, prior work shows that simply increasing test-time compute by extending chain-of-thought (CoT) reasoning does not always yield better results. In this work, we propose three confidence metrics for efficiently estimating model
uncertainty at inference time. We use these metrics to develop AURA (Adaptive Uncertainty-Aware Reasoning
Architecture), a pipeline that dynamically escalates between different CoT lengths based on model confidence. We
evaluate AURA on MedQA, a medical reasoning benchmark, and show that it outperforms state-of-the-art medical
reasoning models such as MedReason by ≈ 4.6% while reducing inference-time token usage by ≈ 55%. These results
demonstrate the promise of uncertainty-aware test-time inference for improving domain-specific LLM performance without
substantially increasing training-time scale.

The figure below depicts the three metrics we use to estimate model confidence:

--FIGURE HERE--

From this, we create a pipeline that uses these metrics to dynamically allocate reasoning:

--FIGURE HERE--


We see a 4.6% increase in accuracy and 55% reduction in token usage on the MedQA dataset when using our pipeline as compared to state-of-the-art baselines such as MedReason.


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
