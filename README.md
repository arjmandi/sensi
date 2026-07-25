# Sensi: Learn One Thing at a Time
**Curriculum-Based Test-Time Learning for LLM Game Agents**

[![arXiv](https://img.shields.io/badge/arXiv-2603.17683-b31b1b.svg)](https://arxiv.org/abs/2603.17683)
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
arXiv preprint, March 2026

```bibtex
@misc{arjmandi2026sensi,
  title={Sensi: Learn One Thing at a Time — Curriculum-Based Test-Time Learning for LLM Game Agents},
  author={Mohsen Arjmandi},
  year={2026},
  eprint={2603.17683},
  archivePrefix={arXiv},
  primaryClass={cs.AI}
}
```

## 📍 Status & Roadmap (March 2026)

- [x] Paper submitted to arXiv (cs.AI + cs.LG)
- [x] Full code + Colab
- [ ] v3 perception fix (hybrid programmatic + LLM diff) → expected +1–2 solves
- [ ] Submit to NeurIPS 2026 Agentic AI / Test-Time Compute workshops

## ➡️ Next step: ARG (July 2026)

Sensi's controlled experiments ended in two single-variable results that pointed past prompting entirely. **Binding loss:** a prose fact and its matching render content never joined across ~2,300 actions of a frozen mid-tier model — until one mechanical coordinate join flipped first-level completions from 0/200-action runs to 3/3 on an otherwise identical stack. **Commitment loss:** a run wrote the winning move verbatim twice and executed it zero times across 171 consecutive turns. Both capabilities existed in the model; neither survived the context. Rhetoric ("pay attention to X") restored neither.

**[ARG — Aligned Referent Grounding](https://github.com/arjmandi/ARG)** is the successor built from those results: an architecture where the LLM *proposes* typed operations through closed vocabularies and admission gates, while a deterministic Executive owns every belief, plan, and achievement — nothing becomes knowledge without a pre-registered prediction matched against observation by code, and everything is append-only and replayable. The goal curriculum is *generated* from typed knowledge deficits starting at zero knowledge; plans compile down to single actions through machine-verified goal chains.

Status there, stated the way that project states everything: the mechanism claims are validated under the system's own run-validity gates (the commitment-drift dissociation at protocol seed count; binding drift structurally absorbed into code; a small backbone running the full loop with a clean write path; generalization to unseen games), while the efficacy claim is open — zero level completions across 52 gated runs, pre-registered as a structural defeater rather than a caveat. The design, campaign record, instruments, tests, and proposal documents are all in the repo.

## 🔗 Connect

- **LinkedIn**: [linkedin.com/in/marjmandi](https://linkedin.com/in/marjmandi)
- **Email**: mohsen.arjmandi@gmail.com
- **Current role**: CTO @ evolutionID (production agent systems + GRID patent)

Built as independent research while leading a PIAM company. Open to collaboration, feedback, or test-time / agent-scaling discussions.

---

⭐ **Star this repo** if you're working on test-time compute, continual learning, neuro-symbolic agents, or ARC-AGI!
