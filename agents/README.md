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





Overview

README.md:1 positions this repo as the ARC-AGI-3 agent starter kit, walking you through env setup, API keys, and how to run or submit agents.
main.py:69 is the CLI entrypoint: it boots logging, loads .env, fetches available games from the ARC backend, and prints the REST root before launching an agent.
agents/__init__.py:19 auto-discovers every Agent subclass (plus JSONL playbacks) so --agent=<name> picks from the registered templates or recordings.
agents/structs.py:62 defines the shared Pydantic models for game frames, scorecards, and the GameAction enum, keeping HTTP payloads and validations consistent.
Core Runtime Flow

main.py:182 creates a Swarm with the chosen agent, games list, and optional tags, then runs it on a background thread while the main thread manages shutdown.
agents/swarm.py:35 opens a remote scorecard, spawns one agent thread per game, joins them, logs the final JSON scorecard, and posts a web URL before cleanup.
agents/agent.py:45 implements the reusable agent loop: persistent requests.Session, recording hooks, action submission via take_action, and configurable stop conditions.
agents/tracing.py:50 wraps each agent’s main in an AgentOps trace when an API key is present, otherwise falling back to a safe no-op tracer.
Agent Implementations

agents/templates/random_agent.py:9 gives the minimal RNG baseline that resets as needed and samples simple/complex actions to illustrate the base API.
agents/templates/llm_agents.py:16 supplies the core LLM agent plus ReasoningLLM, FastLLM, and GuidedLLM variants that control observation cadence, reasoning effort, and metadata logging.
agents/templates/langgraph_functional_agent.py:47 and agents/templates/langgraph_thinking/agent.py:14 show two LangGraph styles: the former a single LLM call, the latter a multi-node workflow with analysis, memory, and image tooling.
agents/templates/reasoning_agent.py:44 and agents/templates/smolagents.py:24 extend the LLM template into structured reasoning, zone-aware vision, and smolagents tool-calling (both text and image modes).
Support & Instrumentation

agents/recorder.py:15 writes every frame, scorecard snapshot, and LLM output to timestamped JSONL files, enabling the Playback agent and external analysis.
agents/agent.py:132 automatically appends validated frames to the recorder, updates the run GUID, and caps runs at MAX_ACTIONS to avoid infinite loops.
agents/swarm.py:117 handles the REST scorecard/open and scorecard/close calls, returning a typed Scorecard that agents can log or persist at shutdown.
agents/templates/langgraph_functional_agent.py:180 and agents/templates/langgraph_thinking/vision.py:20 factor frame pretty-printing and PNG rendering so models can consume grids as formatted text or annotated imagery.
Testing & Next Steps

tests/unit/test_core.py:15 covers GameAction, FrameData, Scorecard, and both random agents; additional suites under tests/unit validate recorder persistence and swarm orchestration.
README.md:46 documents optional AgentOps setup that is already wired into main.py and agents/tracing.py if you supply AGENTOPS_API_KEY.
main.py:171 supports custom tags for each scorecard session, handy for experiment tracking across different prompts or agent tweaks.
Suggested next steps: 1) uv run pytest to confirm the environment, 2) uv run main.py --agent=langgraphfunc --game=<game_id> to observe an advanced template, 3) subclass agents/templates/llm_agents.py to codify your own prompting or tool strategy.
in the llmagent , the openai api is the new responses api or the older completion?


They’re using the classic Chat Completions endpoint. In agents/templates/llm_agents.py:63, the code builds an OpenAI() client and every call goes through client.chat.completions.create(...) (see the observation branch at agents/templates/llm_agents.py:116 and the action branch around agents/templates/llm_agents.py:147). The new Responses API would call client.responses.create(...), so this file is still on the older chat-completions interface.

in the llmagent, other agents like guidedllm use the llm(agent) as paent?


Yes. In agents/templates/llm_agents.py, classes such as ReasoningLLM, FastLLM, GuidedLLM, and the example MyCustomLLM all inherit from LLM (which itself already inherits from Agent). For instance, class GuidedLLM(LLM, Agent): simply layers extra prompting and metadata capture on top of the shared LLM agent loop—no need to reimplement the networking or message bookkeeping that LLM provides.

in python, if llm inherits from agent, why do i need to explicitly mention agent when i create guidedllm from llm


You don’t need to, strictly speaking. LLM already subclasses Agent, so writing class GuidedLLM(LLM, Agent): introduces Agent a second time. Python’s MRO collapses duplicates, so it behaves as if you’d written class GuidedLLM(LLM):, just with extra noise. The extra base is likely a copy‑paste holdover; if you prefer the cleaner form, you can safely drop Agent in those derived classes and rely on the inheritance chain from LLM.

write a highlevel description of chooseaction funciton under llm class


LLM.choose_action orchestrates the full chat loop with the OpenAI model each turn. On the very first call it seeds conversation state by building the user prompt and forcing an initial RESET action. On later turns it logs the previous tool call, optionally gets a narrative “observation” message from the model, then supplies a fresh user prompt and asks the model—either through the Functions or Tools API—for the next game action. The method captures any JSON arguments, falls back to a safe default if parsing fails, and returns a populated GameAction that upstream code submits to the ARC server.

explain it like 1. 2. 3. ...


If this is the very first turn, seed the chat history with a user prompt and pretend the model already chose RESET, then return that action immediately to start the game.
For every subsequent turn, log the previous tool call (the game’s response) back into the conversation so the model can observe what just happened.
Optionally ask the model for a short textual observation about the new frame and keep that in the message history.
Send the latest user prompt plus the accumulated messages to the OpenAI chat endpoint (functions or tools style) and let it pick the next action.
Parse the model’s function/tool call, convert its JSON arguments into a GameAction, and hand that action upstream for execution.
can we say that llm agents are all implementing LLM class with diffferent prompts in the build_prompt fucntion then the choose_action from the LLM class implements a loop with/without observation?


Exactly. Every concrete LLM-based agent subclasses the shared LLM class and customizes its prompting (usually by overriding build_user_prompt, sometimes build_func_resp_prompt or the tool schema). The base choose_action method then runs the standard loop—first seeding with a forced RESET, then, on later turns, replaying the prior tool call, optionally asking the model for an observation, and finally requesting the next action. Flags like DO_OBSERVATION, MODEL_REQUIRES_TOOLS, or REASONING_EFFORT simply dial that loop up or down for each variant.

give me an overview how different llm implementation differ from the llm itself. don't ignore any difference


ReasoningLLM (agents/templates/llm_agents.py:403) – switches to the o4-mini reasoning model, forces tool-calling (MODEL_REQUIRES_TOOLS = True), and overrides choose_action plus track_tokens to stash rich reasoning metadata (tokens, response preview, game context) in action.reasoning. It also adds capture_reasoning_from_response so parents can pull structured token counts from the API payload.
FastLLM (agents/templates/llm_agents.py:472) – keeps the base prompts but disables the observation pass (DO_OBSERVATION = False), so choose_action skips the intermediate “comment on the frame” call and jumps straight to requesting the next tool/function invocation.
GuidedLLM (agents/templates/llm_agents.py:496) – runs on OpenAI’s o3 model with tool calls required, sets REASONING_EFFORT = "high", and shrinks the rolling message window to 10. Like ReasoningLLM, it decorates action.reasoning and track_tokens, but adds custom metadata (agent_type, game_rules) and supplies an extensive, hand-authored prompt describing the LockSmith rules.
MyCustomLLM (agents/templates/llm_agents.py:565) – provided as a scaffold; it inherits all loop mechanics unchanged and simply demonstrates how to overwrite build_user_prompt (and optionally other helpers) to encode your own domain instructions.
in reasoningllmm how it uses tool calling


ReasoningLLM inherits the base loop that always talks to the model via function/tool calls, but it locks the configuration to enforce that behavior. Specifically:

at definition time (agents/templates/llm_agents.py:403) it sets MODEL_REQUIRES_TOOLS = True, so inside LLM.choose_action every OpenAI request routes through the tools payload rather than the legacy functions field.
when the class overrides choose_action, it still calls super().choose_action(...), so the base method handles building the tool schema (build_tools()), posting the conversation to client.chat.completions.create(..., tools=..., tool_choice="required"), and reading the returned tool call.
after the parent returns, ReasoningLLM adds metadata to action.reasoning, but the actual tool invocation and parsing all come from the base logic with MODEL_REQUIRES_TOOLS = True.


