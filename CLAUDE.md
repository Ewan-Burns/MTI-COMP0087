# MTI-COMP0087: Matching Thy Inference

## Project Overview
UCL COMP0087 StatNLP research project investigating how inference/sampling methods affect AI text detection. Building on Dubois et al. (2025) "Your AI is a Troll".

**Research Questions:**
1. What inference methods best evade AI detection?
2. What inference methods best detect AI-generated text?

## Key Files

| File | Purpose |
|------|---------|
| `src/config.py` | All 37 sampling configs, 4 model configs, 4 detector configs |
| `src/generation.py` | Text generation with sampling strategies |
| `src/api_generation.py` | Groq API-based generation (free, for testing) |
| `src/detectors/` | Binoculars, FastDetectGPT, RADAR, RoBERTa implementations |
| `src/metrics.py` | Text quality metrics (MTLD, diversity, etc.) |
| `src/experiment.py` | Main experiment orchestration |
| `PROJECT_PLAN.md` | Detailed research plan with timeline |

## Models
- **Generators:** Llama 3.2 3B, Mistral 7B, Qwen2 7B, Deepseek 7B (optional)
- **Detectors:** Binoculars, FastDetectGPT, RADAR, RoBERTa

## Sampling Methods (37 total)
Greedy, temperature (0.5-1.2), top-k (10-1000), top-p (0.3-0.95), typical (0.3-0.95), eta (1e-4 to 0.1), repetition penalty (1.05-1.30)

## Environment Setup
- HuggingFace token in `.env` as `HF_TOKEN`
- Groq API key in `.env` as `GROQ_API_KEY` (for quick testing)
- Install: `pip install -r requirements.txt`

## Quick Commands
```bash
# Quick test (API-based, no GPU needed)
python -c "from src.api_generation import quick_test; print(quick_test())"

# Run experiment
python scripts/run_experiment.py --mode quick

# Full experiment (requires GPU)
python scripts/run_experiment.py --mode full
```

## Current Status
- Code scaffolding: Complete
- Detectors implemented: 4/4 (Binoculars, FastDetectGPT, RADAR, RoBERTa)
- Sampling configs: 37/37 from Dubois et al.
- Next: Run experiments on GPU machine

## Notes
- Local model loading requires ~16GB+ RAM or GPU with 8GB+ VRAM
- Use Groq API for quick testing without GPU
- See PROJECT_PLAN.md for full experimental protocol
