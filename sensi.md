#Sensi

## Intro
Sensi is an agent designed to win the ARC AGI 3

### Design Boundaries: limitations and assumptions
db1: In Sensi we don't want the agent to learn how many game are there. Agent always will start a game must be able to figure out what's going on and then play
~~db2: Based on db1, the timer in the game is a part of the game to be learned.~~ // there's no timer. it's energy line
db2: in Sensi we train our model to do simple things, like actions, move, etc. (it can also do api calls instead of simple actions which makes the model more general and more complicated)

Rules
r1: if the agent is in tool discovery, it must be able to update the description of the tools it's calling

## Design
1. Sensi must learn what felt good to try and what didn't felt good
2. 

## Bets
1. We bet on implementing the tool calling ourselves to have more control over the logic of what function to call. we will limit the model output with pydantic or dspy, so we can always get the output correctly. but this bet is something that we're not sure and in case of failure it's better to also try giving the model the control for tool use.



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

Minimal pseudo-flow: (not the best example! actually kinda missleading)
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


### A clean game-loop pattern (controller-owned)
1. Send goal + current state summary (+ tools).
2. Assistant returns `tool_calls` (e.g., `act({...})`).
3. You execute the action, compute a **curated observation**, and send it back **as the tool result tied to that** `**tool_call_id**`.
4. Repeat. If you’re on Assistants/Responses, the thread stores these steps for you; on Chat, you resend the running history. [OpenAI Platform+1](https://platform.openai.com/docs/guides/tools?utm_source=chatgpt.com)

