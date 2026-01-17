# SensiLLM v2 Architecture Plan

## Overview
Redesign SensiLLM with a structured learning loop that manages "items to learn" with metrics, scoring, and state transitions before delegating to Player1/Player2.

---

## 1. New Database Schema

### File: `agents/templates/llm_agents.py` (in `SensiLLM.__init__`)

#### Table: `items_to_learn`
Stores learning items with state machine and scoring.

```sql
CREATE TABLE IF NOT EXISTS items_to_learn (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id TEXT,
    card_id TEXT,
    item_name TEXT NOT NULL,
    state TEXT DEFAULT 'not_reached',  -- 'not_reached' | 'learning' | 'fact'
    learning_metric TEXT,               -- LLM-generated verification metric (nullable)
    threshold INTEGER DEFAULT 8,        -- per-item threshold (1-10)
    current_sense_score INTEGER,        -- latest score from SenseScorer (1-10)
    UNIQUE(game_id, card_id, item_name)
)
```

#### Table: `inputs`
Key-value store for frame data, game state, previous moves, etc.

```sql
CREATE TABLE IF NOT EXISTS inputs (
    game_id TEXT,
    card_id TEXT,
    turn_id INTEGER,
    key TEXT,
    value TEXT,  -- JSON-encoded value
    PRIMARY KEY (game_id, card_id, turn_id, key)
)
```

---

## 2. New DSPy Signatures

### File: `agents/templates/llm_agents.py` (after existing signatures)

### 2.1 MetricGenerator Signature
Generates a learning metric for an item.

```python
class MetricGeneratorSignature(dspy.Signature):
    """Given an item to learn, generate a metric to verify learning."""

    item_to_learn: str = dspy.InputField(
        desc="The item/concept the agent needs to learn about the game"
    )

    learning_metric: str = dspy.OutputField(
        desc="A clear description of how to verify that this item has been learned. "
             "What observations or outcomes would confirm understanding?"
    )
```

### 2.2 SenseScorer Signature
Scores how well an item has been learned.

```python
class SenseScorerSignature(dspy.Signature):
    """Score the agent's understanding of a learning item (1-10)."""

    item_to_learn: str = dspy.InputField(
        desc="The item/concept being evaluated"
    )
    learning_metric: str = dspy.InputField(
        desc="The criteria for verifying learning"
    )
    facts: List[str] = dspy.InputField(
        desc="Confirmed facts from previously learned items"
    )
    figured_out: List[str] = dspy.InputField(
        desc="Things Player1 has figured out through gameplay"
    )
    inputs: dict = dspy.InputField(
        desc="Current game inputs (frame_summary, game_state, prev_action, etc.)"
    )

    sense_score: int = dspy.OutputField(
        desc="Score from 1-10 indicating learning progress. "
             "1=no understanding, 10=complete understanding"
    )
    reasoning: str = dspy.OutputField(
        desc="Brief explanation for the score"
    )
```

### 2.3 Modified Player1 Signature
Add `facts` and `current_item_to_learn` inputs.

```python
class Player1(dspy.Signature):
    # Existing inputs...
    current_frame = dspy.InputField()
    prev_frame = dspy.InputField()
    prev_decision_type = dspy.InputField()
    prev_action = dspy.InputField()
    frame_diff = dspy.InputField()
    losing_sequences = dspy.InputField()
    prev_guesses = dspy.InputField()
    prev_figured_out = dspy.InputField()
    guidelines = dspy.InputField()

    # NEW inputs
    facts: List[str] = dspy.InputField(
        desc="Confirmed facts from items marked as learned"
    )
    current_item_to_learn: str = dspy.InputField(
        desc="The item currently being learned"
    )

    # Outputs (unchanged)
    guesses: List[str] = dspy.OutputField()
    figured_out: List[str] = dspy.OutputField()
```

### 2.4 Modified Player2 Signature
Add `facts`, `current_item_to_learn`, and modify prompt.

```python
class Player2(dspy.Signature):
    """Pick an action to make sense of the current item to learn."""

    guesses: List[str] = dspy.InputField()
    figured_out: List[str] = dspy.InputField()

    # NEW inputs
    facts: List[str] = dspy.InputField(
        desc="Confirmed facts from items marked as learned"
    )
    current_item_to_learn: str = dspy.InputField(
        desc="The item currently being learned"
    )
    inputs: dict = dspy.InputField(
        desc="Current game inputs (frame data, game state, etc.)"
    )

    # Outputs (unchanged)
    decision_type = dspy.OutputField()
    action = dspy.OutputField()
```

---

## 3. New Helper Methods

### File: `agents/templates/llm_agents.py` (in `SensiLLM` class)

### 3.1 `get_current_item_to_learn()`
Returns the item currently being learned, or picks the next available one.

```python
def get_current_item_to_learn(self, game_id: str, card_id: str) -> Optional[dict]:
    """
    Returns the current learning item, or None if all items are facts.
    Priority: 1) item with state='learning', 2) first item with state='not_reached'
    """
```

### 3.2 `update_item_state()`
Updates an item's state and scores in the database.

```python
def update_item_state(self, item_id: int, state: str = None,
                      metric: str = None, sense_score: int = None):
    """Update learning item state, metric, or sense_score."""
```

### 3.3 `get_facts()`
Returns all items marked as 'fact'.

```python
def get_facts(self, game_id: str, card_id: str) -> List[str]:
    """Return list of item_names where state='fact'."""
```

### 3.4 `store_input()` / `get_inputs()`
Store and retrieve key-value inputs.

```python
def store_input(self, game_id: str, card_id: str, turn_id: int,
                key: str, value: Any):
    """Store a key-value input for this turn."""

def get_inputs(self, game_id: str, card_id: str, turn_id: int) -> dict:
    """Get all inputs for this turn as a dict."""
```

---

## 4. New Control Flow in `choose_action()`

### File: `agents/templates/llm_agents.py` (method `choose_action`)

```
┌─────────────────────────────────────────────────────────────────┐
│                        choose_action()                          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  1. FIRST FRAME CHECK (existing)                                │
│     if latest_frame.frame == []:                                │
│       → Initialize, store inputs, return RESET                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  2. LOAD PREVIOUS STATE (existing)                              │
│     → load_prev_state_for_player1()                             │
│     → Convert current frame to image                            │
│     → Calculate frame_diff                                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  3. STORE INPUTS (NEW)                                          │
│     → store_input("frame", frame_summary)                       │
│     → store_input("game_state", game_state)                     │
│     → store_input("prev_action", prev_action)                   │
│     → store_input("prev_decision_type", prev_decision_type)     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  4. LEARNING EVALUATION (NEW) - ONE ITEM AT A TIME              │
│                                                                 │
│     current_item = get_current_item_to_learn()                  │
│     (Returns: item with state='learning', or first 'not_reached')│
│                                                                 │
│     if current_item is None:                                    │
│       → All items learned! Proceed to Player1/2 with no target  │
│                                                                 │
│     else:                                                       │
│       # Mark as 'learning' if it was 'not_reached'              │
│       if current_item.state == 'not_reached':                   │
│         → update_item_state(state='learning')                   │
│                                                                 │
│       # Generate metric if missing                              │
│       if current_item.learning_metric is None:                  │
│         ┌──────────────────────────────────────────────┐        │
│         │  4a. METRIC GENERATOR (DSPy)                 │        │
│         │  Input: item_name                            │        │
│         │  Output: learning_metric                     │        │
│         │  → Store metric in DB                        │        │
│         │  → Proceed to Player1/2 (don't score yet)    │        │
│         └──────────────────────────────────────────────┘        │
│                                                                 │
│       else: # Has metric, evaluate sense score                  │
│         ┌──────────────────────────────────────────────┐        │
│         │  4b. SENSE SCORER (DSPy)                     │        │
│         │  Input: item, metric, facts, figured_out,    │        │
│         │         inputs                               │        │
│         │  Output: sense_score (1-10), reasoning       │        │
│         │  → Store sense_score in DB                   │        │
│         └──────────────────────────────────────────────┘        │
│                                                                 │
│         if sense_score >= threshold:                            │
│           → Mark item as 'fact'                                 │
│           → Get next item (becomes new current_item_to_learn)   │
│                                                                 │
│       # Proceed to Player1/2 with current_item_to_learn         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  5. PLAYER 1 - OBSERVER (modified)                              │
│                                                                 │
│     Inputs (existing):                                          │
│       current_frame, prev_frame, prev_decision_type,            │
│       prev_action, frame_diff, losing_sequences,                │
│       prev_guesses, prev_figured_out, guidelines                │
│                                                                 │
│     Inputs (NEW):                                               │
│       facts (from get_facts())                                  │
│       current_item_to_learn                                     │
│                                                                 │
│     Outputs: guesses, figured_out                               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  6. PLAYER 2 - ACTOR (modified)                                 │
│                                                                 │
│     Inputs (existing): guesses, figured_out                     │
│                                                                 │
│     Inputs (NEW):                                               │
│       facts                                                     │
│       current_item_to_learn                                     │
│       inputs (key-value dict)                                   │
│                                                                 │
│     Prompt context:                                             │
│       "Based on facts, guesses, figured_out, pick an action     │
│        to make sense of [current_item_to_learn].                │
│        Your teammate observes your action results."             │
│                                                                 │
│     Outputs: decision_type, action                              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  7. RETURN ACTION (existing)                                    │
│     → Store observation and decision                            │
│     → Return parsed action                                      │
└─────────────────────────────────────────────────────────────────┘
```

---

## 5. Updated Guidelines Prompts

### 5.1 MetricGenerator Prompt
```
You are helping an agent learn about an interactive game environment.

Given an item the agent needs to learn: "{item_to_learn}"

Your task: Describe HOW the agent can verify that it has learned this item.

Think about:
- What observations would confirm understanding?
- What test actions could validate the learning?
- What patterns in game feedback would indicate mastery?

Output a clear, actionable metric that can be used to score learning progress.
```

### 5.2 SenseScorer Prompt
```
You are judging an agent's learning progress on a specific item.

Item to learn: "{item_to_learn}"
Learning metric: "{learning_metric}"

Current knowledge:
- Facts (confirmed): {facts}
- Figured out: {figured_out}
- Game inputs: {inputs}

Based on the learning metric, score the agent's understanding from 1 to 10:
- 1-3: No/minimal understanding
- 4-6: Partial understanding, still exploring
- 7-8: Good understanding, some gaps
- 9-10: Complete understanding, can apply reliably

Output your score and brief reasoning.
```

### 5.3 Modified Player1 Guidelines (additions)
```
(existing guidelines...)

Additionally:
- You have access to "facts" - these are items your team has definitively learned and confirmed.
- You are currently trying to learn: "{current_item_to_learn}"
- Focus your guesses and observations toward understanding this item.
- Use the facts to inform your analysis, but develop guesses about the current learning target.
```

### 5.4 Modified Player2 Guidelines (additions)
```
(existing guidelines...)

Additionally:
- You have access to "facts" - confirmed learnings that you can rely on.
- Current learning target: "{current_item_to_learn}"
- Choose actions that help make sense of this specific item.
- Your teammate (Player 1) will observe the results to progress learning.
```

---

## 6. Implementation Order

1. **Add new database tables** in `SensiLLM.__init__`
   - `items_to_learn` table
   - `inputs` table

2. **Add helper methods** to `SensiLLM` class
   - `get_current_item_to_learn()`
   - `update_item_state()`
   - `get_facts()`
   - `store_input()` / `get_inputs()`

3. **Create new DSPy signatures**
   - `MetricGeneratorSignature`
   - `SenseScorerSignature`

4. **Modify existing DSPy signatures**
   - Add `facts` and `current_item_to_learn` to `Player1`
   - Add `facts`, `current_item_to_learn`, `inputs` to `Player2`
   - Update guidelines strings

5. **Refactor `choose_action()`**
   - Add input storage after loading previous state
   - Add learning loop before Player1 call
   - Pass new parameters to Player1/Player2

6. **Test manually**
   - Add test items to `items_to_learn` table via SQLite
   - Run agent and verify flow

---

## 7. Files to Modify

| File | Changes |
|------|---------|
| `agents/templates/llm_agents.py` | Main implementation - tables, signatures, choose_action flow |

---

## 8. State Machine for Learning Items

```
                    ┌─────────────────┐
                    │   not_reached   │
                    │  (initial state) │
                    └────────┬────────┘
                             │
                     picked as next item
                             │
                             ▼
                    ┌─────────────────┐
        ┌──────────│    learning     │◄────────┐
        │          └────────┬────────┘         │
        │                   │                  │
        │           sense_score evaluated      │
        │                   │                  │
        │     ┌─────────────┴─────────────┐    │
        │     │                           │    │
        │   score >= threshold      score < threshold
        │     │                           │    │
        │     ▼                           └────┘
        │   ┌─────────────────┐          (stay in learning,
        │   │      fact       │           try more actions)
        │   │  (terminal state) │
        │   └─────────────────┘
        │
        └── (all items become facts → game mastery)
```
