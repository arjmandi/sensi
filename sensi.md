#Sensi

## Intro
Sensi is an agent designed to win the ARC AGI 3

### Design Boundaries: limitations and assumptions
db1: In Sensi we don't want the agent to learn how many game are there. Agent always will start a game must be able to figure out what's going on and then play
db2: in Sensi we train our model to do simple things, like actions, move, etc. (it can also do api calls instead of simple actions which makes the model more general and more complicated)

Rules
r1: if the agent is in tool discovery, it must be able to update the description of the tools it's calling

## Design
1. Sensi must learn what felt good to try and what didn't felt good 
1.1. we couldn't articulate that well enough, while we pushed towards the guesses and informed decisions
2. We will simplify the prompts and structure them to reflect game actions and game state with actions and scores, in case of poor performance in guess development, or figured out items we can make thme more comprehensive again

## Bets
1. We bet on implementing the tool calling ourselves to have more control over the logic of what function to call. we will limit the model output with pydantic or dspy, so we can always get the output correctly. but this bet is something that we're not sure and in case of failure it's better to also try giving the model the control for tool use.
2. we bet on using DSPy to minimize the errors in retrieving the objects from the prompts. it shouldn't for now interfere with our prompting, but later we will use the optimization for DSPy to test the performance


## considerations
- we're ignoring available actions from the function response
- 

## important design questions
- when do we decide to put some part of the logic in code, or in the prompt and the line of reasoning?
  - the way we did it previously was that we've abstracted away how to play the game and kept experiences outside the reasoning. 
  - regarding specifically scores, I'm still not sure how to model it:
    - doesn't give away strong signals of winning but to validate guesses about the game
      - so one design change that comes to mind is break down guesses and figured outs into what each action does? how the game is won? then add the score and monitor the score on level change to add this distinction
- should we separate learning of an action from learning the winning? -> it reduced to experiment if figured out items can actually lead to intentional win





level 1
["RESET starts the game", "blue platform (with red top) is our player", "we can move the player in the teal area - red area is not reachable", "ACTION1 moves the player 1 pixel up", "ACTION2 moves the player 1 pixel down", "ACTION3 moves the player 1 pixel left", "ACTION4 moves the player 1 pixel right", "other actions are not needed","each action consumes 1 unit of energy if moves the player, units of energy are show on the top as dots, white means remained energy, teal means consumed energy", "if all dots go teal we can't move and game over", "shape in the middle with blue dot and black wings is the key generator", "the shape in the middle higher than the key generaotr is a door. the purple square with the key shape", "the shape in the left button corner is our current key", "when we pass over the key generator it generates a new key", "we must pass over the key generator until it generates a key matching to the door"]

sample metric for action learning:
"**Criteria:** The learner can accurately describe, predict, and demonstrate the effect of every available action in the game across relevant contexts,  and interactions with game state.

**What to test / observe (judge checklist):**
1. **Action catalog completeness:** Learner enumerates all actions available in the current game (including context-sensitive actions, menu actions, and conditional actions) and groups them by type if applicable (movement, interaction, combat, inventory, UI/meta, etc.).
2. **Correct effect description:** For each action, learner states:
   - Immediate outcome (what changes on screen/state right away)


**Scoring guide (1–10):**
- **1–2:** Knows only a few actions; frequent incorrect descriptions; 
- **3–4:** Identifies at least 4 actions 
- **5–6:** Correct for most actions;
- **7–8:** Accurate for nearly all actions including constraints
- **9:** Complete and precise; 
- **10:** Exhaustive mastery; explains mechanics and feedback signals clearly; "

another good judgement
Criteria to verify the agent has learned what each action does in the game:

1) **Action→Outcome mapping accuracy (core test)**
- For every available action in the action space, the agent can state the expected immediate effects on:
  - the agent’s position/orientation/velocity (if applicable),
  - inventory/equipment/state variables,
  - environment objects (doors, items, enemies, tiles, etc.),
  - turn progression/time cost (if applicable),
  - reward/penalty signals (if any are directly tied to the action).
- Scoring: Across a standardized test suite of scenarios, the agent’s predicted outcomes match the observed outcomes with high accuracy (e.g., ≥90–95% exact match on discrete effects; within tolerance on continuous changes).

2) **Context-conditional behavior understanding**
- The agent correctly predicts when an action:
  - has no effect (e.g., “move into wall”),
  - fails with a specific feedback (e.g., “cannot”, “out of range”, “no ammo”),
  - produces different outcomes depending on context (e.g., “use” on different objects).
- Scoring: In a set of edge-case scenarios, the agent correctly anticipates success/failure and the resulting state change/feedback ≥90% of the time.

3) **Causal intervention validation**
- Given a goal phrased as “cause X to happen” (e.g., “pick up item”, “open door”, “attack enemy”, “advance time”), the agent selects the correct action(s) and demonstrates the expected effect in-game.
- Scoring: For each action, at least one controlled experiment shows the agent can intentionally trigger the action’s signature effect and can also demonstrate a counterexample where the effect does not occur due to missing preconditions.

4) **Generalization to novel states**
- In previously unseen but valid configurations (new room layouts, different object placements, different enemy positions), the agent still predicts and demonstrates the same action semantics.
- Scoring: Maintains ≥85–90% prediction accuracy and correct action selection on held-out scenarios.

5) **Documentation-level completeness**
- The agent can produce a complete “action manual” listing every action, its preconditions, primary effects, failure modes, and any costs (time/energy/resources), consistent with observed gameplay.
- Scoring: A judge can cross-check each entry against gameplay logs; no missing actions, and ≤ minor errors (no major incorrect effect descriptions).

Observations/outcomes that confirm learning:
- High agreement between predicted and actual state transitions after actions.
- Correct handling of edge cases and preconditions.
- Ability to reliably use each action to achieve its intended effect across varied contexts.