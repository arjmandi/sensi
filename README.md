# Sensi: Learn One Thing at a Time
**Curriculum-Based Test-Time Learning for LLM Game Agents**

[![arXiv](https://img.shields.io/badge/arXiv-2603.xxxxx-b31b1b.svg)](https://arxiv.org/abs/2603.xxxxx)
[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org)
[![DSPy](https://img.shields.io/badge/built_with-DSPy-orange.svg)](https://dspy.ai)

**50–94× better sample efficiency on ARC-AGI-3** while openly diagnosing the exact failure mode.

## 🎯 What is Sensi?

Sensi is a neuro-symbolic LLM agent framework that forces the model to **learn one thing at a time** at test time — no retraining, no gradient updates.
It turns the context window into a programmable database and uses an external state machine + dynamic LLM-as-judge to drive curriculum-style learning.

Two iterations:
- **Sensi v1** (two-player Observer/Actor) → solved 2 levels with perfect reproducibility (pass@10 = pass@1)
- **Sensi v2** (full curriculum + SQLite control plane) → solved 0 levels but finished the entire learning curriculum in **~32 turns** (vs 1,600–3,000 reported by baselines)

The paper turns the negative result into a clear contribution: the bottleneck has shifted from "learning efficiency" to "perceptual grounding" — and we show exactly where it breaks (self-consistent hallucination cascade).

## ✨ Key Results & Contributions

- 50–94× sample efficiency improvement on ARC-AGI-3
- Novel **database-as-control-plane** pattern (entire cognitive state lives in SQLite → fully steerable)
- Dynamic LLM-as-judge with generated rubrics + external state machine
- Precise failure diagnosis + actionable next steps (hybrid pixel analysis)
- Full DSPy implementation + reproducible logs

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/getting-started/installation/) package manager
- An [ARC-AGI-3 API key](https://three.arcprize.org/)
- A [Gemini API key](https://aistudio.google.com/)

### Setup & Run

```bash
git clone https://github.com/arjmandi/sensi.git
cd sensi

# Configure API keys
cp .env.example .env
# Edit .env and set ARC_API_KEY and GEMINI_API_KEY

# Install dependencies and run SensiLLM against all games
uv run main.py --agent=sensillm

# Or target specific games
uv run main.py --agent=sensillm --game=ls20
```

📊 **Colab Notebook** (one-click):
[Open in Colab → Sensi v2 Demo](https://colab.research.google.com/github/arjmandi/sensi/blob/main/notebooks/sensi_v2_demo.ipynb)

## 🏗️ Architecture Highlights

- **v1**: Observer + Actor separation (perception vs action)
- **v2**: FrameDiff → MetricGen → SenseScore → Player1 → Player2 pipeline
  + SQLite control plane + curriculum state machine

(See Figure 1 & 4 in the paper for clean diagrams.)

For the research journey from early intuitions to the final design, see [background/BACKGROUND.md](background/BACKGROUND.md).

## 📄 Paper & Citation

**Sensi: Learn One Thing at a Time — Curriculum-Based Test-Time Learning for LLM Game Agents**
Mohsen Arjmandi (CTO, evolutionID)
arXiv preprint (submitted March 2026) — link live within 48 hours

```bibtex
@misc{arjmandi2026sensi,
  title={Sensi: Learn One Thing at a Time — Curriculum-Based Test-Time Learning for LLM Game Agents},
  author={Mohsen Arjmandi},
  year={2026},
  eprint={2603.xxxxx},
  archivePrefix={arXiv},
  primaryClass={cs.AI}
}
```

## 📍 Status & Roadmap (March 2026)

- [x] Paper submitted to arXiv (cs.AI + cs.LG)
- [x] Full code + Colab
- [ ] v3 perception fix (hybrid programmatic + LLM diff) → expected +1–2 solves
- [ ] Submit to NeurIPS 2026 Agentic AI / Test-Time Compute workshops

## 🔗 Connect

- **LinkedIn**: [linkedin.com/in/marjmandi](https://linkedin.com/in/marjmandi)
- **Email**: mohsen.arjmandi@gmail.com
- **Current role**: CTO @ evolutionID (production agent systems + GRID patent)

Built as independent research while leading a PIAM company. Open to collaboration, feedback, or test-time / agent-scaling discussions.

---

⭐ **Star this repo** if you're working on test-time compute, continual learning, neuro-symbolic agents, or ARC-AGI!
