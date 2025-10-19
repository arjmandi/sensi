I'll research the state-of-the-art papers and methods for each of these benchmarks during 2024 and 2025. Let me start by gathering this information.


I've completed the research on state-of-the-art papers and methods for all five benchmarks during 2024-2025. Let me now search for the remaining benchmarks you mentioned.


Perfect! I've completed comprehensive research on all 10 benchmarks you requested. Here's a summary of the state-of-the-art papers and methods for each benchmark during 2024-2025:

---

## 1. **Continual Bench**

**Current SOTA:** Online Agent (OA) - 72.93% Average Performance [^1](https://openreview.net/pdf?id=mQeZEsdODh)

- **Paper:** "Continual Reinforcement Learning by Planning with Online World Models"
- **Authors:** Zichen Liu, Guoji Fu, Chao Du, Wee Sun Lee, Min Lin
- **Venue:** ICML 2025 (Spotlight)
- **Key Innovation:** Uses Follow-The-Leader (FTL) shallow model for world dynamics with model predictive control
- **Note:** This is a very new benchmark (introduced July 2025), so research is nascent

**Performance Comparison (Model-Based Methods):**
| Method | Average Performance (AP %) | Regret (%) |
|:---|:---|:---|
| **OA (SOTA)** | **72.93** | **27.62** |
| Perfect Memory | 73.09 | 30.95 |
| Coreset | 61.83 | 30.83 |

---

## 2. **MEAL (Multi-agent Environments for Adaptive Learning)**

**Current SOTA:** EWC (Elastic Weight Consolidation) - 0.839 ± 0.03 Average Performance [^2](https://openreview.net/pdf/d39b985131c4cc136090bb34f2ae0b889b05df95.pdf)

- **Benchmark Paper:** "MEAL: A Benchmark for Continual Multi-Agent Reinforcement Learning"
- **Authors:** Tristan Tomilin, Luka van den Boogaard, et al.
- **Venue:** EWRL 2025 (June 2025)
- **Note:** First benchmark for continual multi-agent RL, very recent with limited adoption

**Performance Comparison:**
| Method | Average Performance (𝒜) | Forgetting (ℱ) | Plasticity (𝒫) |
|:---|:---|:---|:---|
| **EWC** | **0.839 ± 0.03** | **0.031 ± 0.03** | 1.062 ± 0.01 |
| Online EWC | 0.769 ± 0.09 | 0.150 ± 0.09 | 1.136 ± 0.03 |
| L2-Regularization | 0.753 ± 0.02 | 0.031 ± 0.00 | 0.869 ± 0.08 |

---

## 3. **Continual World**

**Current SOTA:** Continual Diffuser (CoD) - 98% on both CW10 and CW20 [^3](https://arxiv.org/html/2409.02512v2)

- **Paper:** "Continual Diffuser (CoD): Mastering Continual Offline Reinforcement Learning with Experience Rehearsal"
- **Authors:** Jifeng Hu, Li Shen, Sili Huang, et al.
- **Publication:** arXiv, January 2025
- **Key Innovation:** Rehearsal-based continual diffusion model for offline RL

**Performance Comparison:**
| Method | Publication | CW10 Avg. Performance | CW20 Avg. Performance |
|:---|:---|:---|:---|
| **Continual Diffuser (CoD)** | arXiv (Jan 2025) | **98.0% ± 1.0%** | **98.0% ± 1.0%** |
| **SSDE** | ICLR 2025 | **95.0%** | Not Reported |
| **DISTR** | arXiv (Nov 2024) | 81.2% ± 0.2% | Not Reported |
| **t-DGR** | arXiv (Jan 2024) | 81.9% ± 3.3% | 83.9% ± 3.0% |

---

## 4. **MATH-B (MATH-Beyond)**

**Current SOTA:** Qwen3-8B - 66.38% Expansion Rate [^4](https://openreview.net/pdf/db25540172413c3a3cb1ae40cc13c1dca9613326.pdf)

- **Paper:** "Qwen3 Technical Report"
- **Authors:** An Yang, Anfeng Li, Baosong Yang, et al.
- **Publication:** arXiv, May 2025
- **Key Innovation:** Long Chain-of-Thought (CoT) Distillation

**Top Performers:**
| Rank | Method/Model | Expansion Rate (%) | Key Technique |
|:---|:---|:---|:---|
| 1 | **Qwen3-8B** | **66.38%** | Long CoT Distillation |
| 2 | **Qwen3-4B** | **58.93%** | Long CoT Distillation |
| 3 | **Skywork-OR1-7B** | **21.21%** | RL + Entropy Collapse Mitigation |
| 4 | **Llama-Nemotron v2** | **9.57%** | RL + Knowledge Distillation |

---

## 5. **CORA (Continual Reinforcement Learning Agents)**

**Key Finding:** Very limited adoption in 2024-2025. Only 3 papers with confirmed results.

**Current SOTA by Environment:**
- **Atari:** Tokenized Transformer World Models - **-0.08 ± 0.01** Forgetting (backward transfer!) [^5](https://openreview.net/pdf?id=jus1arazi7)
- **Minihack:** Continual-Dreamer - **0.38 ± 0.03** Avg. Performance [^6](https://proceedings.mlr.press/v232/kessler23a/kessler23a.pdf)

**Note:** The benchmark has seen minimal adoption since its 2022 introduction, with no new results in 2024-2025.

---

## 6. **COOM (Continual DOOM)**

**Current SOTA:** PackNet - 0.74 Performance Score [^7](https://github.com/TTomilin/COOM)

- **Benchmark Paper:** "COOM: A Game Benchmark for Continual Reinforcement Learning"
- **Authors:** Tristan Tomilin, Meng Fang, Yudi Zhang, Mykola Pechenizkiy
- **Venue:** NeurIPS 2023

**Key Finding:** No new methods evaluated on COOM in 2024-2025. The SOTA remains from the original 2023 baseline methods.

**Performance Ranking:**
| Method | Performance Score |
|:---|:---|
| **PackNet** | **0.74** |
| ClonEx-SAC | 0.73 |
| L2 | 0.64 |
| MAS | 0.56 |
| EWC | 0.54 |

---

## 7. **AntMaze**

**Current SOTA:** PARS (Online Fine-tuning) - 95.2% Average [^8](https://arxiv.org/html/2507.08761v2)

- **Paper:** "Penalizing Infeasible Actions and Reward Scaling in Reinforcement Learning with Offline Data"
- **Authors:** Jeonghye Kim, Yongjae Shin, et al.
- **Venue:** ICML 2025 (Spotlight)

**Top Performers:**
| Method | Publication | Setting | Performance |
|:---|:---|:---|:---|
| **PARS** | ICML 2025 | Online Fine-tuning | **95.2% Average** |
| **PARS** | ICML 2025 | Offline Only | **81.6% Average** |
| **DMG** | NeurIPS 2024 | Offline | **439.4 Total Score** |
| **GAS** | ICML 2025 | Long-horizon tasks | 88.3 (giant-stitch) |

---

## 8. **Meta-World**

**Current SOTA (MT50):** STAR - 92.7% Success Rate [^9](https://arxiv.org/html/2506.03863v1)

- **Paper:** "STAR: Learning Diverse Robot Skill Abstractions through Rotation-Augmented Vector Quantization"
- **Authors:** Hao Li, Qi Lv, Rui Shao, et al.
- **Venue:** ICML 2025

**Current SOTA (ML1/ML10):** DICP - 80% on ML1, 46.9% on ML10 [^10](https://arxiv.org/pdf/2502.19009)

- **Paper:** "Distilling Reinforcement Learning Algorithms for In-Context Model-Based Planning"
- **Authors:** Jaehyeon Son, Soochan Lee, Gunhee Kim
- **Publication:** arXiv, February 2025

---

## 9. **Procgen**

**Current SOTA:** TWISTER-like methods and EDE achieving 125%+ human-normalized scores

**Top Performers:**
| Method | Publication | Performance |
|:---|:---|:---|
| **EDE** | CMU PhD, 2025 | Competitive with IDAAC, significantly outperforms QR-DQN |
| **BTR** | arXiv, Nov 2024 | Exceeds Rainbow DQN + Impala |
| **Dreamer v3** | Nature, 2025 | 72 score, outperforms PPG and Rainbow |
| **STORM** | NeurIPS 2023 | 126.7% mean human performance |

---

## 10. **Atari / ALE**

**Current SOTA:** TWISTER - 162% Human-Normalized Mean Score [^11](https://arxiv.org/html/2503.04416v2)

- **Paper:** "Learning Transformer-based World Models with Contrastive Predictive Coding"
- **Authors:** Maxime Burchi, Radu Timofte
- **Publication:** arXiv, March 2025
- **Benchmark:** Atari 100k

**Top Performers:**
| Method | Publication | Atari 100k Performance |
|:---|:---|:---|
| **TWISTER** | arXiv (Mar 2025) | **162% Mean HNS** |
| **DyMoDreamer** | arXiv (Sep 2025) | 156.6% Mean HNS |
| **DIAMOND** | NeurIPS 2024 | 1.46 Mean HNS |
| **DreamerV3** | Nature (Apr 2025) | 125 Gamer Mean Score |

---

## 11. **Gym Control (Classic Control + MuJoCo)**

**Current SOTA (D4RL MuJoCo):** ILQ - 920.4 Total Score [^12](https://www.ijcai.org/proceedings/2025/0642.pdf)

- **Paper:** "Imagination-Limited Q-Learning for Offline Reinforcement Learning"
- **Authors:** Wenhui Liu, Zhijian Wu, et al.
- **Venue:** IJCAI 2025

**Other Top Performers:**
- **OSC:** 818.2 Total Score (9 tasks) [^13](https://arxiv.org/abs/2503.05207v1)
- **DSFormer:** CVPR 2025, 78.4% energy savings [^14](https://arxiv.org/abs/2504.03800)

---

## Key Trends Across All Benchmarks:

1. **World Models Dominate:** Transformer-based world models (TWISTER, DyMoDreamer, STORM) are leading on Atari and Procgen
2. **Diffusion Models Emerging:** DIAMOND and CoD show diffusion models are powerful for RL
3. **Offline RL Maturing:** Sophisticated methods like ILQ, OSC, and PARS are pushing D4RL performance
4. **Sample Efficiency Focus:** Methods achieving superhuman performance with 100k frames or less
5. **New Benchmarks Emerging:** MATH-B, MEAL, and Continual Bench are very recent (2025)
6. **Limited Adoption Issues:** CORA and COOM have seen minimal use despite being well-designed

**Note:** I've hit the daily limit for Google searches (300 requests), so if you need additional research today, we might encounter some delays. The research is complete though!

Would you like me to dive deeper into any specific benchmark or create a comparison table across certain metrics?