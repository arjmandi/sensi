#Sensi

## Intro
Sensi is an agent designed to win the ARC AGI 3

### Design Boundaries: limitations and assumptions
db1: In Sensi we don't want the agent to learn how many game are there. Agent always will start a game must be able to figure out what's going on and then play
~~db2: Based on db1, the timer in the game is a part of the game to be learned.~~ // there's no timer. it's energy line
db2: in Sensi we train our model to do simple things, like actions, move, etc. it can also do api calls instead of actions

Rules
r1: if the agent is in tool discovery, it must be able to update the description of the tools it's calling

## Design
1. Sensi must learn what felt good to try and what didn't felt good
2. 




### overal design points

the model **must** see the *new observation/state* after your action—otherwise it’s blind and can’t choose the next move. You don’t have to return the tool’s raw output “as-is,” but you **do** need to feed back a concise state update that the model can reason over.

A solid pattern for a game loop where you add your own thinking:
1. You define tools
* `act({action, target, ...})` → the game engine executes and returns low-level effects.
* (Optional) `observe()` → pulls the *current* game state.
* (Optional) `summarize_state(raw_state)` → **your controller** (not the model) compresses raw state into a clean, model-friendly summary.

2. Controller loop (you own this logic)
* Send the model: (a) system rules (how to interpret the world), (b) the **latest state summary**, and (c) the goal.
* Model replies with a **tool call** to `act(...)`.
* You execute it in the game.
* You **compute an observation** (diff, events, rewards, visible tiles, etc.). You can blend in your “line of thinking” here: filter noise, add hints, safety checks, or a short critique.
* Send that observation back as the **tool result** (or a new user message)—whichever your SDK expects.
* Repeat until done.

Minimal pseudo-flow:
```text
System: You are the game agent. Use the state to pick the best action.
User: Goal = reach the key. State = {position, nearby, inventory, hazards}

Assistant -> tool: act({"action":"move","dir":"north"})
Tool -> assistant: {"applied": true, "events": ["stepped_on:plate"], "raw_state": {...}}

(Controller) summarize raw_state → compact_state

User (or tool message): Observation = {delta: "plate clicked", state: compact_state}

Assistant -> tool: act({"action":"use","item":"grappling_hook"})
...
```

Key tips
* **Always return observations.** They can be: (a) the whole new state, (b) a *diff*, or (c) a curated summary. Choose the smallest form that keeps decisions correct.
* **Inject your own reasoning safely.** Put it in:
  • the **system prompt** (policies, priorities, heuristics), and/or
  • the **observation text** (e.g., “Controller note: the lever likely arms traps”).
  Don’t overwrite facts; mark controller guidance clearly.
* **Structure helps.** Send observations as JSON with stable fields: `{turn, location, visible, inventory, events, goals, constraints}`. The model will be more consistent.
* **Multi-step calls.** If the model returns multiple tool calls, you can execute them in order and then return a single aggregated observation; or (often better) run one action → one observation → next action, to keep feedback tight.

Quick check: in your game, what does a *useful* observation look like after an action (full map? local FOV? event list + reward)? 
