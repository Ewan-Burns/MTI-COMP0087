# MTI-COMP0087: Matching Thy Inference
## Comprehensive Project Plan

### Project Overview

**Research Questions:**
1. What inference methods best evade AI detection?
2. What inference methods best detect AI-generated text?

**Core Hypothesis:** Different sampling/decoding strategies affect both text quality metrics and detector performance. Building on Dubois et al. (2025), we expand the study to multiple models and detectors.

---

## Phase 1: Environment Setup (Complete)

### 1.1 Dependencies
```bash
pip install -r requirements.txt
```

### 1.2 Authentication
- HuggingFace token stored securely in `.env`
- Protected from Git via `.gitignore`

### 1.3 Project Structure
```
MTI-COMP0087/
├── .env                    # HuggingFace token (DO NOT COMMIT)
├── .gitignore              # Protects secrets
├── requirements.txt        # Dependencies
├── PROJECT_PLAN.md         # This file
├── src/
│   ├── __init__.py
│   ├── config.py           # All configurations
│   ├── models.py           # Model loading/caching
│   ├── generation.py       # Text generation
│   ├── metrics.py          # Text quality metrics
│   ├── experiment.py       # Experiment orchestration
│   └── detectors/
│       ├── __init__.py
│       ├── base.py         # Abstract base class
│       ├── binoculars.py   # Binoculars detector
│       ├── fastdetectgpt.py # FastDetectGPT detector
│       ├── radar.py        # RADAR detector
│       └── roberta.py      # RoBERTa detector
├── scripts/
│   └── run_experiment.py   # Example experiment script
├── outputs/                # Experiment outputs (gitignored)
└── Papers/                 # Reference papers
```

---

## Phase 2: Data Preparation

### 2.1 Dataset Selection
**Primary Dataset:** RAID (Realistic AI Detection) or equivalent

**Requirements:**
- Human-written texts (ground truth)
- Diverse domains (news, academic, creative writing)
- Sufficient volume (aim for 1000+ samples per category)

### 2.2 Prompt Extraction
```python
# Example prompt extraction
from datasets import load_dataset

# Load RAID or your dataset
dataset = load_dataset("your_dataset_name")

# Extract prompts (first N words of human texts)
def extract_prompt(text, n_words=20):
    words = text.split()[:n_words]
    return " ".join(words)

prompts = [extract_prompt(t) for t in dataset["train"]["text"]]
```

### 2.3 Recommended Prompt Categories
1. **News Articles** - Factual, formal style
2. **Academic Writing** - Technical, citation-heavy
3. **Creative Fiction** - Narrative, emotional
4. **Social Media** - Informal, conversational
5. **Technical Documentation** - Precise, instructional

---

## Phase 3: Text Generation Experiments

### 3.1 Generator Models
| Model | HuggingFace ID | Priority |
|-------|----------------|----------|
| Llama 3.2 3B | `meta-llama/Llama-3.2-3B-Instruct` | Primary |
| Mistral 7B | `mistralai/Mistral-7B-v0.1` | Primary |
| Qwen2 7B | `Qwen/Qwen2-7B` | Primary |
| Deepseek 7B | `deepseek-ai/deepseek-llm-7b-base` | Optional |

### 3.2 Sampling Configurations (37 total from Dubois et al.)

**Temperature Variations:**
- `temp_0.5`, `temp_0.7`, `temp_0.9`, `temp_1.0`, `temp_1.1`, `temp_1.2`

**Repetition Penalty:**
- `rep_1.05`, `rep_1.10`, `rep_1.15`, `rep_1.20`, `rep_1.25`, `rep_1.30`

**Top-k Sampling:**
- `topk_10`, `topk_20`, `topk_50`, `topk_75`, `topk_100`, `topk_1000`

**Top-p (Nucleus) Sampling:**
- `topp_0.3`, `topp_0.5`, `topp_0.7`, `topp_0.8`, `topp_0.9`, `topp_0.95`

**Typical Sampling:**
- `typical_0.3`, `typical_0.5`, `typical_0.7`, `typical_0.8`, `typical_0.9`, `typical_0.95`

**Eta Sampling:**
- `eta_1e-4`, `eta_1e-3`, `eta_5e-3`, `eta_0.01`, `eta_0.05`, `eta_0.1`

### 3.3 Generation Protocol
```python
from src.config import GENERATOR_MODELS, get_sampling_subset
from src.experiment import Experiment

exp = Experiment(name="main_experiment")

# Run generation for each model
for model_name in ["llama-3.2-3b", "mistral-7b", "qwen2-7b"]:
    exp.run_generation(
        prompts=your_prompts,
        model_configs=[GENERATOR_MODELS[model_name]],
        sampling_configs=get_sampling_subset("dubois_all"),
    )
```

---

## Phase 4: Detection Experiments

### 4.1 Detectors
| Detector | Type | Description |
|----------|------|-------------|
| Binoculars | Unsupervised | Perplexity/cross-entropy ratio |
| FastDetectGPT | Zero-shot | Conditional probability curvature |
| RADAR | Semi-supervised | Adversarially-trained RoBERTa |
| RoBERTa | Supervised | Fine-tuned classifier |

### 4.2 Detection Protocol
```python
from src.config import DETECTOR_CONFIGS

exp.run_detection(
    detector_configs=list(DETECTOR_CONFIGS.values()),
    human_texts=your_human_reference_texts,
)
```

### 4.3 Evaluation Metrics
- **AUROC** - Area under ROC curve (primary metric)
- **TPR@FPR=1%** - True positive rate at 1% false positive rate
- **TPR@FPR=5%** - True positive rate at 5% false positive rate
- **F1 Score** - Harmonic mean of precision and recall

---

## Phase 5: Text Quality Analysis

### 5.1 Metrics (from `src/metrics.py`)

| Metric | Description | Human Reference |
|--------|-------------|-----------------|
| MTLD | Measure of Textual Lexical Diversity | 94.60 |
| Hapax Ratio | Proportion of unique words | 0.3490 |
| Simpson Index | Vocabulary concentration | 0.0066 |
| Zipf Alpha | Word frequency distribution slope | 1.20 |
| Heaps Beta | Vocabulary growth rate | 0.5946 |
| Readability | Flesch-Kincaid Reading Ease | 50.34 |
| Self-BLEU | Diversity across generations | 42.89 |

### 5.2 Key Findings from Dubois et al.
- **Repetition penalty > 1.15** causes detector failure
- **Temperature > 1.1** significantly reduces detectability
- **Greedy decoding** is most detectable
- Settings closest to human statistics perform best

---

## Phase 6: Analysis & Visualization

### 6.1 Required Figures
1. **AUROC Heatmap** - Detector performance vs. sampling config
2. **Diversity vs. Detectability** - Scatter plot with Pareto frontier
3. **Quality Metric Comparison** - Bar charts vs. human reference
4. **Model Comparison** - Cross-model performance analysis

### 6.2 Analysis Questions
1. Which sampling configs evade detection across all detectors?
2. Which detectors are most robust to sampling variations?
3. How do different generator models compare?
4. Is there a quality-detectability trade-off?

---

## Phase 7: Experimental Timeline

### Week 1-2: Setup & Pilot
- [ ] Verify environment setup
- [ ] Run pilot experiments with small sample
- [ ] Debug any issues with model loading
- [ ] Validate metric calculations

### Week 3-4: Generation Experiments
- [ ] Generate texts for Llama 3.2 3B (all 37 configs)
- [ ] Generate texts for Mistral 7B (all 37 configs)
- [ ] Generate texts for Qwen2 7B (all 37 configs)
- [ ] Compute text quality metrics

### Week 5-6: Detection Experiments
- [ ] Run Binoculars on all generated texts
- [ ] Run FastDetectGPT on all generated texts
- [ ] Run RADAR on all generated texts
- [ ] Fine-tune and run RoBERTa detector

### Week 7-8: Analysis & Writing
- [ ] Compute all evaluation metrics
- [ ] Generate visualizations
- [ ] Statistical significance tests
- [ ] Write report/paper

---

## Phase 8: Running Experiments

### Quick Test (Verify Setup)
```bash
cd scripts
python run_experiment.py --mode quick
```

### Full Experiment
```bash
python run_experiment.py --mode full
```

### Custom Experiment
```python
from src.experiment import Experiment
from src.config import GENERATOR_MODELS, SAMPLING_CONFIGS, DETECTOR_CONFIGS

exp = Experiment(name="custom_experiment")

# Your custom configuration
exp.run_generation(
    prompts=your_prompts,
    model_configs=[GENERATOR_MODELS["llama-3.2-3b"]],
    sampling_configs=[
        SAMPLING_CONFIGS["greedy"],
        SAMPLING_CONFIGS["temp_0.9"],
        SAMPLING_CONFIGS["rep_1.20"],
    ],
)

# Run detection
exp.run_detection(
    detector_configs=[DETECTOR_CONFIGS["binoculars"]],
    human_texts=your_human_texts,
)

# Get results
auroc_matrix = exp.compute_auroc_matrix()
report = exp.generate_report()
```

---

## Key Code Usage

### Single Generation Test
```python
from src.generation import quick_generate

result = quick_generate(
    prompt="The future of AI is",
    model_name="llama-3.2-3b",
    sampling_name="temp_0.7",
)
print(result.generated_text)
```

### Single Detection Test
```python
from src.detectors import FastDetectGPTDetector

detector = FastDetectGPTDetector()
result = detector.detect("Your text here...")
print(f"Score: {result.score:.3f}, Prediction: {result.prediction}")
```

### Compute Metrics
```python
from src.metrics import compute_all_metrics

texts = ["text1...", "text2...", "text3..."]
metrics = compute_all_metrics(texts)
print(f"Diversity: {metrics['diversity_mean']:.3f}")
print(f"MTLD: {metrics['mtld_mean']:.1f}")
```

---

## Checkpoints & Deliverables

### Checkpoint 1: Pilot Complete
- [ ] 100 generations per model (quick_test subset)
- [ ] Detection results for 1 detector
- [ ] Preliminary metrics computed

### Checkpoint 2: Full Generation
- [ ] All 37 configs x 3 models x 1000 prompts
- [ ] All quality metrics computed
- [ ] Results saved to JSON

### Checkpoint 3: Full Detection
- [ ] All 4 detectors evaluated
- [ ] AUROC matrix computed
- [ ] Human reference comparison

### Final Deliverable
- [ ] Complete analysis
- [ ] All visualizations
- [ ] Written report

---

## Troubleshooting

### Common Issues

**CUDA Out of Memory:**
```python
# Use 4-bit quantization in config
model_config.load_in_4bit = True
```

**Model Access Denied:**
- Ensure HuggingFace token is valid
- Accept model license on HuggingFace website
- Check `.env` file is properly formatted

**Slow Generation:**
- Reduce batch_size
- Use smaller model for debugging
- Enable Flash Attention 2 if supported

---

## References

1. Dubois et al. (2025) - "Your AI is a Troll: How Inference Methods Fool Detectors"
2. Hans et al. (2024) - "Spotting LLMs with Binoculars"
3. Bao et al. (2024) - "Fast-DetectGPT"
4. Hu et al. (2023) - "RADAR: Robust AI-Text Detection"
5. Drayson et al. (2025) - Preventing model collapse paper
