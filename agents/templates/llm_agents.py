
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

import json
import logging
import os
import textwrap
from typing import Any, Optional
import re
import openai
from openai import OpenAI as OpenAIClient

from ..agent import Agent
from ..structs import FrameData, GameAction, GameState
import dspy
from typing import List
from pydantic import BaseModel, Field
from dspy import Refine
from dspy.functional import TypedPredictor

# --------------------------------------------------------------------------------------
# Optional: Configure your LM once here (or do it in your app bootstrap)
# --------------------------------------------------------------------------------------

def configure_llm(model: str = "openai/gpt-4o-mini") -> None:
    try:
        dspy.settings.configure(lm=dspy.LM(model, cache=True))
    except Exception:
        # In case DSPy is configured elsewhere or the model alias differs.
        pass

# Call this on import by default (safe if configured elsewhere).
configure_llm()

logger = logging.getLogger()


class LLM(Agent):
    """An agent that uses a base LLM model to play games."""

    MAX_ACTIONS: int = 80
    DO_OBSERVATION: bool = True
    REASONING_EFFORT: Optional[str] = None
    MODEL_REQUIRES_TOOLS: bool = False

    MESSAGE_LIMIT: int = 10
    MODEL: str = "gpt-4o-mini"
    messages: list[dict[str, Any]]
    token_counter: int

    _latest_tool_call_id: str = "call_12345"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.messages = []
        self.token_counter = 0

    @property
    def name(self) -> str:
        obs = "with-observe" if self.DO_OBSERVATION else "no-observe"
        sanitized_model_name = self.MODEL.replace("/", "-").replace(":", "-")
        name = f"{super().name}.{sanitized_model_name}.{obs}"
        if self.REASONING_EFFORT:
            name += f".{self.REASONING_EFFORT}"
        return name

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        """Decide if the agent is done playing or not."""
        return any(
            [
                latest_frame.state is GameState.WIN,
                # uncomment below to only let the agent play one time
                # latest_frame.state is GameState.GAME_OVER,
            ]
        )

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        """Choose which action the Agent should take, fill in any arguments, and return it."""

        logging.getLogger("openai").setLevel(logging.CRITICAL)
        logging.getLogger("httpx").setLevel(logging.CRITICAL)

        client = OpenAIClient(api_key=os.environ.get("OPENAI_API_KEY", ""))

        functions = self.build_functions()
        tools = self.build_tools()

        # if latest_frame.state in [GameState.NOT_PLAYED]:
        if len(self.messages) == 0:
            # have to manually trigger the first reset to kick off agent
            user_prompt = self.build_user_prompt(latest_frame)
            message0 = {"role": "user", "content": user_prompt}
            self.push_message(message0)
            if self.MODEL_REQUIRES_TOOLS:
                message1 = {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": self._latest_tool_call_id,
                            "type": "function",
                            "function": {
                                "name": GameAction.RESET.name,
                                "arguments": json.dumps({}),
                            },
                        }
                    ],
                }
            else:
                message1 = {
                    "role": "assistant",
                    "function_call": {"name": "RESET", "arguments": json.dumps({})},  # type: ignore
                }
            self.push_message(message1)
            action = GameAction.RESET
            return action

        # let the agent comment observations before choosing action
        # on the first turn, this will be in response to RESET action
        function_name = latest_frame.action_input.id.name
        function_response = self.build_func_resp_prompt(latest_frame)
        if self.MODEL_REQUIRES_TOOLS:
            message2 = {
                "role": "tool",
                "tool_call_id": self._latest_tool_call_id,
                "content": str(function_response),
            }
        else:
            message2 = {
                "role": "function",
                "name": function_name,
                "content": str(function_response),
            }
        self.push_message(message2)

        if self.DO_OBSERVATION:
            logger.info("Sending to Assistant for observation...")
            try:
                create_kwargs = {
                    "model": self.MODEL,
                    "messages": self.messages,
                }
                if self.REASONING_EFFORT is not None:
                    create_kwargs["reasoning_effort"] = self.REASONING_EFFORT
                response = client.chat.completions.create(**create_kwargs)
            except openai.BadRequestError as e:
                logger.info(f"Message dump: {self.messages}")
                raise e
            self.track_tokens(
                response.usage.total_tokens, response.choices[0].message.content
            )
            message3 = {
                "role": "assistant",
                "content": response.choices[0].message.content,
            }
            logger.info(f"Assistant: {response.choices[0].message.content}")
            self.push_message(message3)

        # now ask for the next action
        user_prompt = self.build_user_prompt(latest_frame)
        message4 = {"role": "user", "content": user_prompt}
        self.push_message(message4)

        name = GameAction.ACTION5.name  # default action if LLM doesnt call one
        arguments = None
        message5 = None

        if self.MODEL_REQUIRES_TOOLS:
            logger.info("Sending to Assistant for action...")
            try:
                create_kwargs = {
                    "model": self.MODEL,
                    "messages": self.messages,
                    "tools": tools,
                    "tool_choice": "required",
                }
                if self.REASONING_EFFORT is not None:
                    create_kwargs["reasoning_effort"] = self.REASONING_EFFORT
                response = client.chat.completions.create(**create_kwargs)
            except openai.BadRequestError as e:
                logger.info(f"Message dump: {self.messages}")
                raise e
            self.track_tokens(response.usage.total_tokens)
            message5 = response.choices[0].message
            logger.debug(f"... got response {message5}")
            tool_call = message5.tool_calls[0]
            self._latest_tool_call_id = tool_call.id
            logger.debug(
                f"Assistant: {tool_call.function.name} ({tool_call.id}) {tool_call.function.arguments}"
            )
            name = tool_call.function.name
            arguments = tool_call.function.arguments

            # sometimes the model will call multiple tools which isnt allowed
            extra_tools = message5.tool_calls[1:]
            for tc in extra_tools:
                logger.info(
                    "Error: assistant called more than one action, only using the first."
                )
                message_extra = {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": "Error: assistant can only call one action (tool) at a time. default to only the first chosen action.",
                }
                self.push_message(message_extra)
        else:
            logger.info("Sending to Assistant for action...")
            try:
                create_kwargs = {
                    "model": self.MODEL,
                    "messages": self.messages,
                    "functions": functions,
                    "function_call": "auto",
                }
                if self.REASONING_EFFORT is not None:
                    create_kwargs["reasoning_effort"] = self.REASONING_EFFORT
                response = client.chat.completions.create(**create_kwargs)
            except openai.BadRequestError as e:
                logger.info(f"Message dump: {self.messages}")
                raise e
            self.track_tokens(response.usage.total_tokens)
            message5 = response.choices[0].message
            function_call = message5.function_call
            logger.debug(f"Assistant: {function_call.name} {function_call.arguments}")
            name = function_call.name
            arguments = function_call.arguments

        if message5:
            self.push_message(message5)
        action_id = name
        if arguments:
            try:
                data = json.loads(arguments) or {}
            except Exception as e:
                data = {}
                logger.warning(f"JSON parsing error on LLM function response: {e}")
        else:
            data = {}

        action = GameAction.from_name(action_id)
        action.set_data(data)
        return action

    def track_tokens(self, tokens: int, message: str = "") -> None:
        self.token_counter += tokens
        if hasattr(self, "recorder") and not self.is_playback:
            self.recorder.record(
                {
                    "tokens": tokens,
                    "total_tokens": self.token_counter,
                    "assistant": message,
                }
            )
        logger.info(f"Received {tokens} tokens, new total {self.token_counter}")
        # handle tool to debug messages:
        # with open("messages.json", "w") as f:
        #     json.dump(
        #         [
        #             msg if isinstance(msg, dict) else msg.model_dump()
        #             for msg in self.messages
        #         ],
        #         f,
        #         indent=2,
        #     )

    def push_message(self, message: dict[str, Any]) -> list[dict[str, Any]]:
        """Push a message onto stack, store up to MESSAGE_LIMIT with FIFO."""
        self.messages.append(message)
        if len(self.messages) > self.MESSAGE_LIMIT:
            self.messages = self.messages[-self.MESSAGE_LIMIT :]
        if self.MODEL_REQUIRES_TOOLS:
            # cant clip the message list between tool and tool_call else llm will error
            while (
                self.messages[0].get("role")
                if isinstance(self.messages[0], dict)
                else getattr(self.messages[0], "role", None)
            ) == "tool":
                self.messages.pop(0)
        return self.messages

    def build_functions(self) -> list[dict[str, Any]]:
        """Build JSON function description of game actions for LLM."""
        empty_params: dict[str, Any] = {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        }
        functions: list[dict[str, Any]] = [
            {
                "name": GameAction.RESET.name,
                "description": "Start or restart a game. Must be called first when NOT_PLAYED or after GAME_OVER to play again.",
                "parameters": empty_params,
            },
            {
                "name": GameAction.START.name,
                "description": "Start a game. Must be called first when NOT_PLAYED or after GAME_OVER to play again.",
                "parameters": empty_params,
            },
            {
                "name": GameAction.ACTION1.name,
                "description": "Send this simple input action (1, W, Up).",
                "parameters": empty_params,
            },
            {
                "name": GameAction.ACTION2.name,
                "description": "Send this simple input action (2, S, Down).",
                "parameters": empty_params,
            },
            {
                "name": GameAction.ACTION3.name,
                "description": "Send this simple input action (3, A, Left).",
                "parameters": empty_params,
            },
            {
                "name": GameAction.ACTION4.name,
                "description": "Send this simple input action (4, D, Right).",
                "parameters": empty_params,
            },
            {
                "name": GameAction.ACTION5.name,
                "description": "Send this simple input action (5, Enter, Spacebar, Delete).",
                "parameters": empty_params,
            },
            {
                "name": GameAction.ACTION6.name,
                "description": "Send this complex input action (6, Click, Point).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "x": {
                            "type": "string",
                            "description": "Coordinate X which must be Int<0,63>",
                        },
                        "y": {
                            "type": "string",
                            "description": "Coordinate Y which must be Int<0,63>",
                        },
                    },
                    "required": ["x", "y"],
                    "additionalProperties": False,
                },
            },
        ]
        return functions

    def build_tools(self) -> list[dict[str, Any]]:
        """Support models that expect tool_call format."""
        functions = self.build_functions()
        tools: list[dict[str, Any]] = []
        for f in functions:
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": f["name"],
                        "description": f["description"],
                        "parameters": f.get("parameters", {}),
                        "strict": True,
                    },
                }
            )
        return tools

    def build_func_resp_prompt(self, latest_frame: FrameData) -> str:
        return textwrap.dedent(
            """
# State:
{state}

# Score:
{score}

# Frame:
{latest_frame}

# TURN:
Reply with a few sentences of plain-text strategy observation about the frame to inform your next action.
        """.format(
                latest_frame=self.pretty_print_3d(latest_frame.frame),
                score=latest_frame.score,
                state=latest_frame.state.name,
            )
        )

    def build_user_prompt(self, latest_frame: FrameData) -> str:
        """Build the user prompt for the LLM. Override this method to customize the prompt."""
        return textwrap.dedent(
            """
# CONTEXT:
You are an agent playing a dynamic game. Your objective is to
WIN and avoid GAME_OVER while minimizing actions.

One action produces one Frame. One Frame is made of one or more sequential
Grids. Each Grid is a matrix size INT<0,63> by INT<0,63> filled with
INT<0,15> values.

# TURN:
Call exactly one action.
        """.format()
        )

    def pretty_print_3d(self, array_3d: list[list[list[Any]]]) -> str:
        lines = []
        for i, block in enumerate(array_3d):
            lines.append(f"Grid {i}:")
            for row in block:
                lines.append(f"  {row}")
            lines.append("")
        return "\n".join(lines)

    def cleanup(self, *args: Any, **kwargs: Any) -> None:
        if self._cleanup:
            if hasattr(self, "recorder") and not self.is_playback:
                meta = {
                    "llm_user_prompt": self.build_user_prompt(self.frames[-1]),
                    "llm_tools": self.build_tools()
                    if self.MODEL_REQUIRES_TOOLS
                    else self.build_functions(),
                    "llm_tool_resp_prompt": self.build_func_resp_prompt(
                        self.frames[-1]
                    ),
                }
                self.recorder.record(meta)
        super().cleanup(*args, **kwargs)


class ReasoningLLM(LLM, Agent):
    """An LLM agent that uses o4-mini and captures reasoning metadata in the action.reasoning field."""

    MAX_ACTIONS = 80
    DO_OBSERVATION = True
    MODEL_REQUIRES_TOOLS = True
    MODEL = "o4-mini"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._last_reasoning_tokens = 0
        self._last_response_content = ""
        self._total_reasoning_tokens = 0

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        """Override choose_action to capture and store reasoning metadata."""

        action = super().choose_action(frames, latest_frame)

        # Store reasoning metadata in the action.reasoning field
        action.reasoning = {
            "model": self.MODEL,
            "action_chosen": action.name,
            "reasoning_tokens": self._last_reasoning_tokens,
            "total_reasoning_tokens": self._total_reasoning_tokens,
            "game_context": {
                "score": latest_frame.score,
                "state": latest_frame.state.name,
                "action_counter": self.action_counter,
                "frame_count": len(frames),
            },
            "response_preview": self._last_response_content[:200] + "..."
            if len(self._last_response_content) > 200
            else self._last_response_content,
        }

        return action

    def track_tokens(self, tokens: int, message: str = "") -> None:
        """Override to capture reasoning token information from reasoning models."""
        super().track_tokens(tokens, message)

        # Store the response content for reasoning context (avoid empty or JSON strings)
        if message and not message.startswith("{"):
            self._last_response_content = message
        self._last_reasoning_tokens = tokens
        self._total_reasoning_tokens += tokens

    def capture_reasoning_from_response(self, response: Any) -> None:
        """Helper method to capture reasoning tokens from OpenAI API response.

        This should be called from the parent class if we have access to the raw response.
        For reasoning models, reasoning tokens are in response.usage.completion_tokens_details.reasoning_tokens
        """
        if hasattr(response, "usage") and hasattr(
            response.usage, "completion_tokens_details"
        ):
            if hasattr(response.usage.completion_tokens_details, "reasoning_tokens"):
                self._last_reasoning_tokens = (
                    response.usage.completion_tokens_details.reasoning_tokens
                )
                self._total_reasoning_tokens += self._last_reasoning_tokens
                logger.debug(
                    f"Captured {self._last_reasoning_tokens} reasoning tokens from {self.MODEL} response"
                )


class FastLLM(LLM, Agent):
    """Similar to LLM, but skips observations."""

    MAX_ACTIONS = 80
    DO_OBSERVATION = False
    MODEL = "gpt-4o-mini"

    def build_user_prompt(self, latest_frame: FrameData) -> str:
        return textwrap.dedent(
            """
# CONTEXT:
You are an agent playing a dynamic game. Your objective is to
WIN and avoid GAME_OVER while minimizing actions.

One action produces one Frame. One Frame is made of one or more sequential
Grids. Each Grid is a matrix size INT<0,63> by INT<0,63> filled with
INT<0,15> values.

# TURN:
Call exactly one action.
        """.format()
        )


class GuidedLLM(LLM):
    """Similar to LLM, with explicit human-provided rules in the user prompt to increase success rate."""

    MAX_ACTIONS = 80
    DO_OBSERVATION = True
    MODEL = "o3"
    MODEL_REQUIRES_TOOLS = True
    MESSAGE_LIMIT = 10
    REASONING_EFFORT = "high"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._last_reasoning_tokens = 0
        self._last_response_content = ""
        self._total_reasoning_tokens = 0

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        """Override choose_action to capture and store reasoning metadata."""

        action = super().choose_action(frames, latest_frame)

        # Store reasoning metadata in the action.reasoning field
        action.reasoning = {
            "model": self.MODEL,
            "action_chosen": action.name,
            "reasoning_effort": self.REASONING_EFFORT,
            "reasoning_tokens": self._last_reasoning_tokens,
            "total_reasoning_tokens": self._total_reasoning_tokens,
            "game_context": {
                "score": latest_frame.score,
                "state": latest_frame.state.name,
                "action_counter": self.action_counter,
                "frame_count": len(frames),
            },
            "agent_type": "guided_llm",
            "game_rules": "locksmith",
            "response_preview": self._last_response_content[:200] + "..."
            if len(self._last_response_content) > 200
            else self._last_response_content,
        }

        return action

    def track_tokens(self, tokens: int, message: str = "") -> None:
        """Override to capture reasoning token information from o3 models."""
        super().track_tokens(tokens, message)

        # Store the response content for reasoning context (avoid empty or JSON strings)
        if message and not message.startswith("{"):
            self._last_response_content = message
        self._last_reasoning_tokens = tokens
        self._total_reasoning_tokens += tokens

    def capture_reasoning_from_response(self, response: Any) -> None:
        """Helper method to capture reasoning tokens from OpenAI API response.

        This should be called from the parent class if we have access to the raw response.
        For o3 models, reasoning tokens are in response.usage.completion_tokens_details.reasoning_tokens
        """
        if hasattr(response, "usage") and hasattr(
            response.usage, "completion_tokens_details"
        ):
            if hasattr(response.usage.completion_tokens_details, "reasoning_tokens"):
                self._last_reasoning_tokens = (
                    response.usage.completion_tokens_details.reasoning_tokens
                )
                self._total_reasoning_tokens += self._last_reasoning_tokens
  



# --------------------------------------------------------------------------------------
# Structured output for the update step
# --------------------------------------------------------------------------------------

class TurnUpdate(BaseModel):
    guesses: List[str] = Field(
        default_factory=list,
        description="Updated guesses, each a single concise sentence."
    )
    tryings: List[str] = Field(
        default_factory=list,
        description="Concrete experiments/actions to run next, imperative voice."
    )
    figured_out: List[str] = Field(
        default_factory=list,
        description="Proven rules/regularities distilled from successful tryings."
    )

class UpdateSignature(dspy.Signature):
    """
    You are a curious teenager playing a vintage pixel-graphics puzzle.
    You cannot see the screen; you receive frames as arrays of color codes.

    Each turn you get:
      (1) a previous frame snapshot,
      (2) a current frame snapshot,
      (3) your last move,
      (4) the diff describing what changed.

    You maintain three lists:
      - guesses,
      - tryings ,
      - figured_out 

    Update your guesses/tryings/figured_out based on last action + diff.
    Keep items short, specific, and non-duplicated.
    """

    state: str = dspy.InputField(desc="High-level game state and goals.")
    prev_frame: str = dspy.InputField(desc="Previous frame as printed grid.")
    current_frame: str = dspy.InputField(desc="Current frame as printed grid.")
    prev_action: str = dspy.InputField(desc="Action taken in the previous turn.")
    diff: str = dspy.InputField(desc="Observed changes since the last action.")

    prev_guesses: str = dspy.InputField(desc="Previous guesses, free text or bullets.")
    prev_tryings: str = dspy.InputField(desc="Previous tryings, free text or bullets.")
    prev_figured_out: str = dspy.InputField(desc="Previous figured_out, free text or bullets.")

    plan: TurnUpdate = dspy.OutputField(
        desc="Updated lists: guesses, tryings, figured_out."
    )

class TurnPlanner(dspy.Module):
    def __init__(self, max_items: int = 8):
        super().__init__()
        self.predict = TypedPredictor(UpdateSignature)
        self.max_items = max_items

        def reward(args, pred):
            p: TurnUpdate | None = getattr(pred, "plan", None)
            if p is None:
                return 0.0
            lists = [p.guesses, p.tryings, p.figured_out]
            types_ok = all(isinstance(x, list) for x in lists)
            length_ok = all(len(x) <= self.max_items for x in lists)
            non_empty_items = all(all(isinstance(it, str) and it.strip() for it in x) for x in lists)
            return 1.0 if (types_ok and length_ok and non_empty_items) else 0.0

        self.refine = Refine(self.predict, N=3, threshold=1.0, reward_fn=reward)

    def __call__(
        self,
        state: str,
        prev_frame: str,
        current_frame: str,
        prev_action: str,
        diff: str,
        prev_guesses: str | List[str],
        prev_tryings: str | List[str],
        prev_figured_out: str | List[str],
    ) -> TurnUpdate:
        # Normalize list/str inputs to strings (Signature expects str for prev_*)
        def to_text(x):
            if isinstance(x, list):
                return "\n".join(f"- {s}" for s in x)
            return x or ""

        pred = self.refine(
            state=state,
            prev_frame=prev_frame,
            current_frame=current_frame,
            prev_action=prev_action,
            diff=diff,
            prev_guesses=to_text(prev_guesses),
            prev_tryings=to_text(prev_tryings),
            prev_figured_out=to_text(prev_figured_out),
        )
        plan: TurnUpdate = pred.plan

        # Cleanup: uniqueness, trimming, and cap length
        def clean(xs: List[str]) -> List[str]:
            seen, out = set(), []
            for s in xs:
                s = s.strip()
                key = s.lower()
                if s and key not in seen:
                    seen.add(key)
                    out.append(s)
                if len(out) >= self.max_items:
                    break
            return out

        plan.guesses = clean(plan.guesses)
        plan.tryings = clean(plan.tryings)
        plan.figured_out = clean(plan.figured_out)
        return plan

# --------------------------------------------------------------------------------------
# Action selection step
# --------------------------------------------------------------------------------------

class ActionChoice(BaseModel):
    action_name: str = Field(
        description="One of: RESET, ACTION1, ACTION2, ACTION3, ACTION4, ACTION5, ACTION6, ACTION7, START."
    )
    args: Dict[str, Any] = Field(default_factory=dict, description="Optional parameters for the chosen action.")
    rationale: Optional[str] = Field(default=None, description="Why this action is best.")

class ActionSignature(dspy.Signature):
    """
    You are a curious teenager playing a vintage pixel-graphics puzzle.
    You cannot see the screen; you receive frames as arrays of color codes.
    In previous turn you've written these
      - guesses, 
      - tryings,
      - figured_out 

    based on the guesses, tryings and figured out things choose the next action.
    Output only one action.
    """
    state: str = dspy.InputField(desc="High-level game state and goals.")
    latest_frame: str = dspy.InputField(desc="Current frame as printed grid.")
    last_action: str = dspy.InputField(desc="Most recent action taken.")
    diff: str = dspy.InputField(desc="Observed changes caused by the last action.")
    guesses: List[str] = dspy.InputField(desc="UPDATED guesses.")
    tryings: List[str] = dspy.InputField(desc="UPDATED tryings.")
    figured_out: List[str] = dspy.InputField(desc="UPDATED figured_out.")

    choice: ActionChoice = dspy.OutputField(desc="The next action to execute.")

class ActionPlanner(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = TypedPredictor(ActionSignature)

    def __call__(
        self,
        state: str,
        latest_frame: str,
        last_action: str,
        diff: str,
        guesses: List[str],
        tryings: List[str],
        figured_out: List[str],
    ) -> ActionChoice:
        pred = self.predict(
            state=state,
            latest_frame=latest_frame,
            last_action=last_action,
            diff=diff,
            guesses=guesses,
            tryings=tryings,
            figured_out=figured_out,
        )
        choice: ActionChoice = pred.choice
        choice.action_name = (choice.action_name or "").strip().upper()
        return choice

# --------------------------------------------------------------------------------------
# Mapping helper: model → enum + payload
# --------------------------------------------------------------------------------------

# NOTE: we import at runtime to avoid circulars if your project structure differs.
def to_game_action(choice: ActionChoice) -> Tuple[Any, Any]:
    """
    Map ActionChoice to your (enum_member, payload) pair.

    Returns:
        (enum_member, payload)
    Where:
        enum_member is a GameAction member defined in your codebase.
        payload is a tuple or dict you adapt to your SimpleAction/ComplexAction constructors.

    Adjust this to your actual action classes, if you want to instantiate them here.
    """
    # Delayed import to avoid hard dependency if this module is imported in tooling contexts.
    from enum import Enum
    try:
        # Expect GameAction to be importable from your codebase. Adjust path if needed.
        from .game_types import GameAction  # example: project-specific location
    except Exception:
        # Fallback: try a flat import (same dir)
        try:
            from game_types import GameAction  # type: ignore
        except Exception:
            GameAction = None  # type: ignore

    name = (choice.action_name or "").strip().upper()
    if GameAction is None or not hasattr(GameAction, "__members__"):
        raise RuntimeError(
            "GameAction enum not found. Ensure it is importable in llm_agents.py (adjust import above)."
        )

    if name not in GameAction.__members__:
        raise ValueError(f"Model returned invalid action '{name}'. Valid: {list(GameAction.__members__.keys())}")

    enum_member = GameAction.__members__[name]

    # Payload choice: pass through args. If your engine expects specific types,
    # create them here. Example:
    #   if name == "ACTION6": payload = ComplexAction(**choice.args)
    #   else: payload = SimpleAction(**choice.args)
    payload = choice.args or {}

    return enum_member, payload

# --------------------------------------------------------------------------------------
# SensiLLMDS — ties both steps into your agent loop
# --------------------------------------------------------------------------------------

class SensiLLMDS:
    """
    A two-step DSPy flow:
      update (guesses/tryings/figured_out) → choose action
    """

    # You can override these if your app provides them elsewhere.
    MODEL: str = "openai/gpt-4o-mini"
    REASONING_EFFORT: str = "medium"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # These attributes are expected/used by choose_action().
        self.game_state: Any = kwargs.get("game_state", {})
        self.last_action: Any = kwargs.get("last_action", "NONE")
        self.messages: List[Any] = kwargs.get("messages", [])
        self.action_counter: int = kwargs.get("action_counter", 0)

        # Rolling lists we maintain each turn:
        self.guesses: List[str] = kwargs.get("guesses", [])
        self.tryings: List[str] = kwargs.get("tryings", [])
        self.figured_out: List[str] = kwargs.get("figured_out", [])

        # Helpers for frame printing and diff; override by injection if needed.
        self.pretty_print_3d = kwargs.get("pretty_print_3d", lambda f: str(f))

        # Keep a rolling textual snapshot of the last frame so UpdateSignature has prev/current.
        self._prev_frame_text: Optional[str] = None

        # Planners
        self._planner: Optional[TurnPlanner] = None
        self._action_planner: Optional[ActionPlanner] = None

    # ---- Step 1: update lists
    def compute_turn_update(self, latest_frame) -> TurnUpdate:
        planner = self._planner or TurnPlanner(max_items=8)
        self._planner = planner

        current_frame_text = self.pretty_print_3d(latest_frame.frame)
        prev_frame_text = self._prev_frame_text or "(none)"

        # Basic diff synthesis, replace with your richer diff if available on latest_frame/self
        try:
            state_name = getattr(latest_frame.state, "name", str(latest_frame.state))
        except Exception:
            state_name = str(getattr(latest_frame, "state", "UNKNOWN"))
        diff_text = f"score={getattr(latest_frame, 'score', 'NA')}, state={state_name}"

        updated = planner(
            state=str(self.game_state),
            prev_frame=prev_frame_text,
            current_frame=current_frame_text,
            prev_action=str(self.last_action),
            diff=diff_text,
            prev_guesses=self.guesses,
            prev_tryings=self.tryings,
            prev_figured_out=self.figured_out,
        )

        # Persist for the next turn
        self.guesses = updated.guesses
        self.tryings = updated.tryings
        self.figured_out = updated.figured_out
        self._prev_frame_text = current_frame_text
        return updated

    # ---- Step 2: decide action using updated lists
    def choose_action(self, frames: List[Any], latest_frame: Any):
        # Bootstrap on first call: return RESET and seed previous frame.
        if len(self.hypthesis) == 0:
            self._prev_frame_text = self.pretty_print_3d(latest_frame.frame)
            return GameAction.RESET

        # 1) Update lists
        updated = self.compute_turn_update(latest_frame)

        # 2) Choose action
        action_planner = self._action_planner or ActionPlanner()
        self._action_planner = action_planner

        current_frame_text = self.pretty_print_3d(latest_frame.frame)
        try:
            state_name = getattr(latest_frame.state, "name", str(latest_frame.state))
        except Exception:
            state_name = str(getattr(latest_frame, "state", "UNKNOWN"))
        diff_text = f"score={getattr(latest_frame, 'score', 'NA')}, state={state_name}"

        choice = action_planner(
            state=str(self.game_state),
            latest_frame=current_frame_text,
            last_action=str(self.last_action),
            diff=diff_text,
            guesses=updated.guesses,
            tryings=updated.tryings,
            figured_out=updated.figured_out,
        )

        enum_member, payload = to_game_action(choice)

        # Telemetry/trace (optional): attach for your logs, not required by engine.
        self.last_reasoning = {
            "model": self.MODEL,
            "action_chosen": getattr(enum_member, "name", str(enum_member)),
            "reasoning_effort": self.REASONING_EFFORT,
            "game_context": {
                "score": getattr(latest_frame, "score", "NA"),
                "state": state_name,
                "action_counter": self.action_counter,
                "frame_count": len(frames),
            },
            "updated": updated.model_dump(),
            "planner_choice": choice.model_dump(),
        }

        # Caller can now send (enum_member, payload) to the game server.
        # return enum_member, payload, updated
        return enum_member
