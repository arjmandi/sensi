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

## 🚀 Quick Start & Demo

```bash
git clone https://github.com/mohsenarjmandi/Sensi.git
cd Sensi
pip install -r requirements.txt

# Run the full 32-turn curriculum demo (Gemini 3.1 Pro or ChatGPT 5.1)
python run_sensi_v2.py --game LS20 --demo
```

📊 **Colab Notebook** (one-click):
[Open in Colab → Sensi v2 Curriculum Run](https://colab.research.google.com/github/mohsenarjmandi/Sensi/blob/main/notebooks/Sensi_v2_Demo.ipynb)

## 🏗️ Architecture Highlights

- **v1**: Observer + Actor separation (perception vs action)
- **v2**: FrameDiff → MetricGen → SenseScore → Player1 → Player2 pipeline
  + SQLite control plane + curriculum state machine

(See Figure 1 & 4 in the paper for clean diagrams.)

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
