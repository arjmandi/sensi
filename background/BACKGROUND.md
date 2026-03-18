# Background: The Ideas Behind Sensi

This document traces the research journey that led to the Sensi agent — from early
intuitions about perception and model-building, through a survey of continual RL,
to the two-player cooperative design that became Sensi v1 and the curriculum-based
learning system in v2.

---

## 1. Core Intuition: Sensing Before Acting

The starting premise was simple: **an intelligent system must sense before it acts.**

"Sensi" began as a thought experiment — what if an agent first builds a compressed
model of what it observes, then reasons about actions, rather than mapping inputs
to outputs end-to-end?

Key questions from the early notes:

- *What is the closest compressed model to what I'm seeing?*
- *How much weight should we give to memory vs. new data?*
- *Can we build a fully online model that uses an LLM as its "eyes"?*

The insight was that LLMs already encode rich world knowledge. If we treat the LLM
as a perception and reasoning engine — not a policy — we can build learning on top
without touching model weights.

## 2. Models as First-Class Objects

A significant portion of the early research cataloged different mathematical models
(64 model families, from linear functions through Neural ODEs and discrete
differential geometry) to answer: **what kind of model should an agent build?**

The conclusion was pragmatic:

- A *simple task* is one model (input → output mapping).
- A *complex task* is a composition of models.
- Intelligence is the ability to quickly search whether existing learned models
  can be composed into a solution for a new problem.

For ARC-AGI-3, the "model" turned out to be the LLM itself — but the agent's job
is to feed it the right context and extract structured outputs, not to learn new
weights.

## 3. Continual RL Survey: Plasticity vs. Stability

A literature survey of 29+ continual RL papers (2018–2025) across 11 benchmarks
(Continual World, Meta-World, Procgen, Atari, AntMaze, etc.) revealed a consistent
pattern:

**SOTA methods (2024–2025) use adaptive partial updates or hybrid models —
enabling plasticity without catastrophic forgetting.**

Key takeaways that shaped Sensi's design:

- **World models dominate**: TWISTER, Continual Diffuser, DyMoDreamer all use
  learned world models as perception layers before policy learning.
- **Selective updates beat full retraining**: EWC, PackNet, and parameter-free
  optimizers (Fast TRAC) show that you don't need to retrain everything.
- **Online learning is hard but possible**: The "Online Agent" method on
  ContinualBench (72.93% AP) combines follow-the-leader with model-predictive
  control.

This led to the design principle: **freeze the LLM weights, externalize all
learning state into a database, and build a curriculum on top.**

## 4. The Two-Player Design

The breakthrough was reframing game-playing as a cooperative task between two
"players" communicating through structured lists:

**Player 1 (Observer):**
- Sees the game frames (before and after each action)
- Maintains a `guesses` list (hypotheses about the game)
- Maintains a `figured_out` list (confirmed knowledge)
- Analyzes frame diffs to update both lists

**Player 2 (Actor):**
- Reads Player 1's lists
- Chooses actions: either testing a guess (GUESS) or acting on confirmed
  knowledge (INFORMED)
- One action per turn

This separation mirrors the scientific method: observe → hypothesize → test →
confirm. The two lists are the only communication channel, which forces the system
to be explicit about what it knows vs. what it's guessing.

### Confidence Stages

The early design identified four natural stages of gameplay:

1. **No clue** — no guesses, no figured-out items
2. **Exploring** — some guesses about actions and environment
3. **Forming strategy** — many guesses, some figured-out, ideas about winning
4. **Executing** — actions, environment, and win condition all figured out

Most games can be won reliably at stage 4. Some can be won earlier with lucky
guesses. The agent's job is to move through these stages efficiently.

## 5. From v1 to v2: Adding a Curriculum

**Sensi v1** implemented the two-player design directly:
- DSPy signatures for Player 1 (observer) and Player 2 (actor)
- Frame-to-image conversion for multimodal analysis
- Frame diff computation via LLM
- Solved 2 ARC-AGI-3 levels with perfect reproducibility

**Sensi v2** added structured learning on top:
- **Items to learn** with a state machine (`not_reached` → `learning` → `completed` → `fact`)
- **Dynamic metric generation** — LLM generates rubrics for each learning item
- **Sense scoring** — LLM-as-judge evaluates understanding (1–10 scale)
- **SQLite control plane** — all cognitive state externalized into a database
- Completed the full learning curriculum in ~32 turns (50–94× fewer than baselines)

The key failure in v2 was perceptual: the LLM hallucinated consistent but wrong
frame descriptions (self-consistent hallucination cascade). The learning system
worked — it just learned wrong facts from wrong perceptions.

## 6. Key Design Principles

These principles emerged from the research and are embedded in the implementation:

1. **Learn one thing at a time** — curriculum-based, not shotgun exploration
2. **Externalize all state** — SQLite as the control plane, not the context window
3. **Separate perception from action** — two-player split prevents conflation
4. **Hypothesize, don't optimize** — guesses and figured-out lists, not reward
   maximization
5. **LLM as oracle, not policy** — use the model's knowledge, don't retrain it
6. **Curate, don't accumulate** — actively prune guesses, resolve contradictions

---

*For the full analysis, see the [paper](../sensipaper/sensi-paper.pdf).*
