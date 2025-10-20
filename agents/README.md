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



• Execution Flow

  - main.py:4-39 loads .env.example then .env, builds ROOT_URL
    from SCHEME/HOST/PORT, and prepares default API headers
    with ARC_API_KEY.
  - main.py:69-178 configures logging to stdout and logs.log,
    parses CLI args, and ensures --agent matches a key from
    agents.__init__.py:19-30 where every Agent subclass (plus
    playback files) is registered in AVAILABLE_AGENTS.
  - main.py:117-158 fetches the playable game ids from GET
    {ROOT_URL}/api/games; for playback agents it falls back to
    deriving the game id prefix from the recording filename.
  - main.py:150-177 filters the game list if --game is
    provided, collects optional --tags, and then initializes
    AgentOps tracing (agents/tracing.py:37-109) if
    AGENTOPS_API_KEY is set.
  - main.py:182-188 instantiates Swarm, which looks up the
    selected agent class, clones the shared requests.Session,
    and augments the tags (agents/swarm.py:35-68).
  - main.py:188-198 starts a daemon thread executing run_agent,
    while the main thread installs a SIGINT handler that points
    to cleanup.
  - Inside the worker thread, run_agent calls Swarm.main() and
    then sends SIGINT to the process to trigger central cleanup
    once work finishes (main.py:42-45).
  - Swarm.main() opens a scorecard via POST {ROOT_URL}/api/
    scorecard/open, instantiates one agent per game with shared
    cookies, and runs each agent on its own daemon thread
    (agents/swarm.py:69-99).
  - Each agent instance executes Agent.main (agents/
    agent.py:74-99), which loops until is_done (implemented
    by the concrete agent) or MAX_ACTIONS is hit, choosing
    actions, posting them to /api/cmd/{action} (agents/
    agent.py:134-153), validating responses, recording frames,
    and emitting logs.
  - When agents finish, Swarm.main() closes the scorecard
    (agents/swarm.py:101-115,138-159), logs the final report,
    calls each agent’s cleanup (recording final scorecards,
    closing sessions; agents/agent.py:177-199), and returns.
  - The SIGINT fired by run_agent invokes cleanup in the
    main thread (main.py:47-66), which ensures any lingering
    scorecard is closed and logs the access URL before exiting.

  Natural next steps: 1) Run python main.py --agent <agent_key>
  once your API server is reachable; 2) Tail logs.log to watch
  per-action telemetry.

• Agent Base Class

  - agents/agent.py:45 __init__: wires up per-agent state (ids, score frames, tracing
    tags), starts a recorder if live play, and builds a requests.Session seeded with
    shared cookies + API headers.
  - agents/agent.py:74 main: core loop (wrapped by trace_agent_session) that keeps
    picking actions until is_done or MAX_ACTIONS, submitting them and logging FPS/score
    as frames arrive, then calls cleanup.
  - agents/agent.py:92 state: convenience property returning the latest
    FrameData.state.
  - agents/agent.py:96 score: convenience property returning the latest score.
  - agents/agent.py:100 seconds: wall-clock runtime rounded to 2 decimals since the
    loop started.
  - agents/agent.py:104 fps: derived actions-per-second metric using action_counter and
    elapsed time.

  Agent Utilities

  - agents/agent.py:111 is_playback: flag telling whether this instance is the Playback
    subclass.
  - agents/agent.py:115 name: builds a recording/tracing name like
    <game_id>.<agent_class_lower>.
  - agents/agent.py:120 start_recording: opens a Recorder; playback agents reuse the
    filename, live agents get a new prefixed file.
  - agents/agent.py:127 append_frame: updates local frame history, captures run GUIDs,
    and records frames to disk when appropriate.
  - agents/agent.py:134 do_action_request: sends the chosen action to {ROOT_URL}/api/
    cmd/{action} with card/game/guid metadata, logging API errors returned in JSON.
  - agents/agent.py:155 take_action: wraps do_action_request, turning the response into
    a validated FrameData; returns None if validation fails so the loop can continue
    safely.

  Scorecard & Cleanup Hooks

  - agents/agent.py:165 get_scorecard: fetches the per-game scorecard snapshot for
    post-run recording or cleanup.
  - agents/agent.py:177 cleanup: one-time teardown that finalizes recordings with
    scorecard data, logs action counts/FPS, and closes the HTTP session.
  - agents/agent.py:200 is_done: abstract—concrete agents decide when gameplay should
    stop based on accumulated frames.
  - agents/agent.py:205 choose_action: abstract—concrete agents pick the next
    GameAction and populate its payload.

  Playback Overrides

  - agents/agent.py:214 __init__: builds a Recorder tied to the source filename, loads
    recorded actions (guarding errors) so playback can mirror a past run.
  - agents/agent.py:232 filter_actions: trims the recording stream down to entries
    containing the serialized action_input packets.
  - agents/agent.py:236 is_done: playback ends when every recorded action has been
    replayed.
  - agents/agent.py:240 choose_action: rehydrates the next recorded action, rebinds its
    game_id, restores reasoning blobs, and throttles to PLAYBACK_FPS.
  - agents/agent.py:269 append_frame: overrides base behavior to avoid double-
    recording; still tracks frames and GUIDs.

  Natural follow-ups: 1) Inspect a specific concrete agent (e.g., agents/templates/
  random_agent.py) to see its is_done/choose_action implementations; 2) Run a playback
  recording to observe how recorded actions drive the loop.


