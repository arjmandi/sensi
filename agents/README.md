# ARC-AGI-3 Agents

For all information on how to build, test, and run agents, as well as the technical specifications of the agent and game APIs, please see the [agents documentation](https://three.arcprize.org/docs#agent-quickstart).


 Agent Loop

  - Each agent derives from Agent, which keeps a running history of FrameData objects in self.frames; the main loop repeatedly calls your choose_action implementation and then posts it to the server
  (agents/agent.py:74-163).
  - FrameData mirrors what the backend returns: grid snapshot, score, state, last action echoed back, currently allowed moves, etc. (agents/structs.py:207-218).

  Inspect The Board

  - Grab the freshest frame inside choose_action with latest_frame = frames[-1].
  - The actual board pixels live in latest_frame.frame, a 3‑dimensional list indexed [channel][row][col]; coordinates are 0‑63, matching the valid x/y range for complex actions.
  - Other useful fields: latest_frame.state (e.g. GameState.NOT_FINISHED vs GameState.WIN), latest_frame.score, latest_frame.available_actions (what the server says is legal next), and
  latest_frame.action_input if you need to see what you just asked the server to do (agents/structs.py:207-215).

  Send An Action

  - Return one of the enum instances from GameAction in choose_action. Simple actions just set action = GameAction.ACTION3; complex actions (currently only ACTION6) should call action.set_data({"x": …,
  "y": …}) before returning (agents/structs.py:122-159).
  - Optional: attach free-form metadata for debugging by setting action.reasoning = "why you chose this".
  - The base class will serialize action.action_data, add game_id/card_id/guid, and POST to /api/cmd/{action.name} (agents/agent.py:134-153). The server responds with the next FrameData, which the loop
  appends to self.frames.

  Putting It Together

  from agents.agent import Agent
  from agents.structs import GameAction, GameState, FrameData

  class MyAgent(Agent):
      def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
          return latest_frame.state in {GameState.WIN, GameState.GAME_OVER}

      def choose_action(self, frames: list[FrameData], latest_frame: FrameData) -> GameAction:
          if latest_frame.state in {GameState.NOT_PLAYED, GameState.GAME_OVER}:
              return GameAction.RESET  # start a new run

          board = latest_frame.frame  # inspect pixels here
          # ... decide what to do ...
          action = GameAction.ACTION6
          action.set_data({"x": 12, "y": 34})
          action.reasoning = "Moving to the interesting tile"
          return action

  Natural next steps:

  1. Flesh out your decision logic based on the board tensor and available_actions.
  2. Run 
