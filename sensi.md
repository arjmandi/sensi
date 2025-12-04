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
1.1. we couldn't articulate that well enough, while we pushed towards the guesses and informed decisions


## Bets
1. We bet on implementing the tool calling ourselves to have more control over the logic of what function to call. we will limit the model output with pydantic or dspy, so we can always get the output correctly. but this bet is something that we're not sure and in case of failure it's better to also try giving the model the control for tool use.
2. we bet on using DSPy to minimize the errors in retrieving the objects from the prompts. it shouldn't for now interfere with our prompting, but later we will use the optimization for DSPy to test the performance


## next
- [ ] the problem with dspy and litellm seems not a real problem for now to be able to focus on other parts we've left it open
- [ ] since I'm introducing new variables to store between game states, I need to dissect the current code to add this functionality in a way that I understand it fully. otherwise I need to write the entire logic for running the agents and calling the apis.
- [ ] frame data format is garbage! if the AI model is multi-modal, it's better to reconstruct the frame and feed it like image
- [ ] using gemini will make a big difference
- [ ] 

## considerations
- we're ignoring available actions from the function response
- 

## important design questions
- when do we decide to put some part of the logic in code, or in the prompt and the line of reasoning?
  - the way we did it previously was that we've abstracted away how to play the game and kept experiences outside the reasoning. 
  - regarding specifically scores, I'm still not sure how to model it:
    - doesn't give away strong signals of winning but to validate guesses about the game
      - so one design change that comes to mind is break down guesses and figured outs into what each action does? how the game is won? then add the score and monitor the score on level change to add this distinction
- 
- 




in case we want more precise stage definition
class EstimateStage(dspy.Signature):
    """Infer confidence stage (1–4) from counts/quality of guesses vs figured_out.
    Return JSON: {"stage": 1|2|3|4, "why": "≤15 words"}.
    Prefer higher stages when figured_out is strong and guesses are few."""
    guesses = dspy.InputField()
    figured_out = dspy.InputField()
    json_out = dspy.OutputField()

class ChooseAction(dspy.Signature):
    """Given stage, frames, diff, last move, and Player 1 lists, pick exactly one action.
    Return JSON: {"decision_type": "GUESS"|"INFORMED", "action": "ACTION1|ACTION2|ACTION3|ACTION4|ACTION5|ACTION7|RESET", "note": "≤20 words"}.
    Trust figured_out > guesses; avoid actions known to cause loss."""
    stage = dspy.InputField()
    current_frame = dspy.InputField()
    previous_frame = dspy.InputField()
    last_move = dspy.InputField()
    diff = dspy.InputField()
    guesses = dspy.InputField()
    figured_out = dspy.InputField()
    guidelines = dspy.InputField()
    json_out = dspy.OutputField()










---


### overall design points

the model **must** see the *new observation/state* after your action—otherwise it’s blind and can’t choose the next move. You don’t have to return the tool’s raw output “as-is,” but you **do** need to feedback a concise state update that the model can reason over.

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

