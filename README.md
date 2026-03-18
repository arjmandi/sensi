# SensiLLM

A multimodal learning agent for [ARC-AGI-3](https://three.arcprize.org/) that uses a two-player cooperative strategy with persistent learning across game turns.

For the design and research behind this agent, see [sensi.md](sensi.md) and the paper in [sensipaper/](sensipaper/).

## How it works

SensiLLM splits game-playing into two cooperative roles:

- **Player 1 (Observer)**: Sees the game frames, analyzes differences between turns, and maintains two lists — `guesses` (hypotheses) and `figured_out` (confirmed knowledge).
- **Player 2 (Actor)**: Reads Player 1's lists and chooses the next action to either test a guess or act on confirmed knowledge.

A learning system tracks what the agent still needs to learn, scores its understanding via a sense scorer, and promotes confirmed guesses into persistent facts stored in SQLite.

## Setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then:

```bash
cp .env.example .env
```

Set your API keys in `.env`:
- `ARC_API_KEY` — from [three.arcprize.org](https://three.arcprize.org/)
- `GEMINI_API_KEY` — from [Google AI Studio](https://aistudio.google.com/)

## Run

```bash
uv run main.py --agent=sensillm
```

Filter to specific games:

```bash
uv run main.py --agent=sensillm --game=ls20,pls5
```

## Architecture

```
agents/
├── agent.py        # Base Agent class (game loop, API calls, recording)
├── sensi_llm.py    # SensiLLM agent + DSPy signatures
├── structs.py      # Data models (GameAction, FrameData, GameState, Scorecard)
├── swarm.py        # Multi-agent orchestration (one thread per game)
├── recorder.py     # JSONL recording/playback
└── tracing.py      # Optional AgentOps observability
```

## License

This project is licensed under the MIT License.
