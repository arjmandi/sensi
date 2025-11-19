
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import inspect
import sys
import types
import json
import logging
import os
import textwrap
import re
import openai
from openai import OpenAI as OpenAIClient
import sqlite3
import textwrap
from typing import ClassVar, List

#
# Compatibility shim for DSPy + LiteLLM.
# DSPy>=2.5 expects `litellm.Router(retry_policy=RetryPolicy(...))`, but the
# version of LiteLLM pinned in this project (1.9.4) exposes `Router` without
# a `retry_policy` parameter and has no `RetryPolicy` helper.  This shim
# patches LiteLLM at import time so that DSPy can import and construct
# `Router` without errors, while still delegating to the original Router
# implementation under the hood.
#
try:
    import litellm
    import litellm.router as _litellm_router

    # Patch Router to accept `retry_policy` if the installed LiteLLM is older.
    _router = getattr(litellm, "Router", None)
    if callable(_router):
        router_sig = inspect.signature(_router)
        if "retry_policy" not in router_sig.parameters:

            class _RetryPolicy:
                def __init__(self, **kwargs: Any) -> None:
                    for key, value in kwargs.items():
                        setattr(self, key, value)

            BaseRouter = _router

            class _PatchedRouter(BaseRouter):  # type: ignore[misc]
                def __init__(
                    self,
                    *args: Any,
                    retry_policy: Optional["_RetryPolicy"] = None,
                    **kwargs: Any,
                ) -> None:
                    num_retries = kwargs.pop("num_retries", None)
                    if num_retries is None and retry_policy is not None:
                        retries = [
                            getattr(retry_policy, "TimeoutErrorRetries", 0),
                            getattr(retry_policy, "RateLimitErrorRetries", 0),
                            getattr(
                                retry_policy,
                                "InternalServerErrorRetries",
                                0,
                            ),
                        ]
                        num_retries = max(retries) if any(retries) else 0

                    kwargs.pop("retry_policy", None)
                    super().__init__(*args, num_retries=num_retries or 0, **kwargs)

                # Drop the `cache` kwarg that DSPy passes through Router into
                # LiteLLM; the old LiteLLM version does not understand it and
                # would forward it all the way to the OpenAI client.
                def completion(
                    self,
                    model: str,
                    messages: List[Dict[str, Any]],
                    **kwargs: Any,
                ):
                    kwargs.pop("cache", None)
                    return super().completion(model=model, messages=messages, **kwargs)

                async def acompletion(
                    self,
                    model: str,
                    messages: List[Dict[str, Any]],
                    **kwargs: Any,
                ):
                    kwargs.pop("cache", None)
                    return await super().acompletion(model=model, messages=messages, **kwargs)

            litellm.Router = _PatchedRouter  # type: ignore[assignment]
            _litellm_router.Router = _PatchedRouter
            if not hasattr(_litellm_router, "RetryPolicy"):
                _litellm_router.RetryPolicy = _RetryPolicy

    # Patch caching so `from litellm.caching import Cache` (used by DSPy)
    # works even on older LiteLLM versions that don't support the
    # `disk_cache_dir` / `type=\"disk\"` API.
    _OrigCache = getattr(litellm, "Cache", None)
    if _OrigCache is not None and callable(_OrigCache):

        class _CompatCache(_OrigCache):  # type: ignore[misc]
            def __init__(
                self,
                *args: Any,
                disk_cache_dir: Optional[str] = None,
                type: str = "local",
                **kwargs: Any,
            ) -> None:
                # Map new-style arguments onto the older Cache API.
                kwargs.pop("disk_cache_dir", None)
                cache_type = kwargs.pop("type", type)
                if cache_type == "disk":
                    cache_type = "local"
                super().__init__(*args, type=cache_type, **kwargs)

        caching_mod = types.ModuleType("litellm.caching")
        caching_mod.Cache = _CompatCache
        sys.modules["litellm.caching"] = caching_mod
except Exception:
    # If anything goes wrong, fall back to LiteLLM's default behavior.
    pass

from ..agent import Agent
from ..structs import FrameData, GameAction, GameState

# Ensure DSPy/dsp cache stays inside the project workspace so we
# don't rely on a writable home directory (important in sandboxed runs).
os.environ.setdefault("DSP_CACHEDIR", os.path.join(os.getcwd(), "dsp_cache"))

import dspy
from pydantic import BaseModel, Field, field_validator
from dspy import Predict

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

    def is_won(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
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

class SensiLLM(LLM):
    """Similar to LLM, with more senses."""
    MAX_ACTIONS = 20
    DO_OBSERVATION = False
    MODEL = "gpt-5"
    MESSAGE_LIMIT = 10
    REASONING_EFFORT = "low"

    conn = sqlite3.connect("agent_state.db")

    cur = conn.cursor()
    cur.execute("CREATE TABLE IF NOT EXISTS guesses (id INTEGER PRIMARY KEY, game_id TEXT, card_id TEXT, guess TEXT)")
    cur.execute("CREATE TABLE IF NOT EXISTS figured_outs (id INTEGER PRIMARY KEY, game_id TEXT, card_id TEXT, figs TEXT)")
    cur.execute("CREATE TABLE IF NOT EXISTS losing_actions_seqs (id INTEGER PRIMARY KEY, game_id TEXT, card_id TEXT, losing_seq TEXT)")
    cur.execute("CREATE TABLE IF NOT EXISTS game (game_id TEXT, game_state TEXT, prev_action TEXT, prev_decision_type TEXT, prev_frame BLOB, card_id TEXT, losing_seq TEXT, frame_diff TEXT)")

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._last_reasoning_tokens = 0
        self._last_response_content = ""
        self._total_reasoning_tokens = 0

    def choose_action(
            self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        """Choose which action the Agent should take, fill in any arguments, and return it."""

        action = GameAction.RESET  # default action if LLM doesnt call one

        # -------------------------------------- Ask LLM what to do ---------------------------------------------
        # current_frame =
        # prev_frame =
        # prev_decision_type =
        # prev_action =
        # frame_diff =
        # losing_sequences =
        # prev_guesses =
        # prev_figured_out =
        #
        # logger.info("Sending to Assistant for action...")
        #
        # player1 = dspy.Predict(Player1)
        # observations = player1(
        #     current_frame=,
        #     prev_frame = ,
        #     prev_decision_type = ,
        #     prev_action = ,
        #     frame_diff = ,
        #     losing_sequences = ,
        #     prev_guesses = ,
        #     prev_figured_out =
        # )




        return action




class Player1(dspy.Signature):
    """Given the inputs, Return two lists: guesses and figured_out.
    Follow the provided guidelines strictly. Output only valid Python lists (no extra text)."""

    PLAYER1_GUIDELINES: ClassVar[str] = textwrap.dedent("""
    You're playing a vintage pixel-graphics puzzle along with your friend. You are on the same team. You're Player 1 and he is Player 2. Player 1 sees the game screen, Player 2 performs actions. You two work like a team very well.

    To play as a team you two have come up with a simple tactic. Player 1 (you), who sees the game and what each action does, maintains two lists:
    1) a list of "guesses"
    2) a list of "figured_out" things.
    Both lists will be visible to Player 2.

    Usually we have 4 stages of confidence in any game:
    stage 1: when we have no guess, no clue what each action does.
    stage 2: once we figure out the actions and a little about the game environment.
    stage 3: we have a lot of guesses about the game environment and some guesses on how to win.
    stage 4: we’ve figured out the actions, we’ve figured out the environment, and we’ve figured out how to win.

    Most reliably, a game can be won while in stage 4, but some games can be won in stages 3, 2, or even 1 due to being lucky and guessing the right thing early.

    Player 1 (you), in each turn receives:
    1. A snapshot of the screen as the current frame: [row][column][color code]
    2. Previous frame in the same format
    3. Previous type of decision Player 2 has done: GUESS or INFORMED
    4. Previous action Player 2 has done: ACTION1, ACTION2, ACTION3, ACTION4, ACTION5, ACTION7, RESET
    5. Diff of frames to help identify changed areas
    6. Losing action sequences: list of sequences of actions that led to game over in previous attempts

    You, Player 1, populate the "guesses" list and "figured_out" list as below:
    1. Develop guesses by asking: "If this action made this change, then this action is what?" and write it in the "guesses" list.
       Consider:
         1) the last action
         2) the changes in the frames
         3) previous guesses
         4) previous figured_out things
         5) previous sequences that led to game over
       Then write simple guesses.
    2. Write all guesses you can make. For example, if you guess an area of the screen is showing a character, if there's a point counter, if there's a timer, etc. Be creative about guesses.
    3. If, based on the last action, you have figured out what each action does, move guesses about that to the "figured_out" list. Write them as simple actions. For example, "ACTION1 jumps over things".
    4. If, based on the last action, you have figured out things about the game environment or how to win, put them in "figured_out". For example, "taking the character to the door makes us win".
    5. If you want your friend, Player 2, to further try an action to see if that makes you progress or lose, add it to the "guesses" list. When your friend is not certain which move to pick, he tries more guesses.
    6. If, based on current figured_out items, you have further guesses, add them to your "guesses" list for next rounds.
    7. Remove guesses that seem unlikely at this point. You have to forget useless guesses; otherwise everything will be a guess and the list becomes useless.
    8. Develop guesses about "things that make you lose" and write them in the "guesses" list. Again, consider:
         1) the last action
         2) the changes in the frames
         3) previous guesses
         4) previous figured_out things
         5) previous sequences that led to game over
    9. If, based on the last action, you can say some guesses definitely make you lose, move them to "figured_out". Write simple sentences for this. Your friend, Player 2, relies on "figured_out" items to avoid things that make you lose.
    10. Remove things you deem unlikely to make you lose from the "guesses" list.
    11. Review the "figured_out" list and if things contradict each other, make a decision and provide a sane list to Player 2.

    You can only communicate with Player 2 through these lists. Be patient. The more you develop "guesses", the more Player 2 will do actions outside "figured_out". The more you develop “figured_out” items and remove guesses, the more Player 2 will play using "figured_out" instead of exploring guesses, meaning reaching higher stages of confidence.

    So help him with smart “guesses” and certain "figured_out" things. Be patient with the list. Player 2 only has one action at a time but you can play as many times as you want. You play action by action to figure out the game and then win it.
    """).strip()

    # --- Inputs ---
    current_frame = dspy.InputField()
    prev_frame = dspy.InputField()
    prev_action_type = dspy.InputField()
    prev_action = dspy.InputField()
    diff = dspy.InputField()
    losing_sequences = dspy.InputField()
    previous_guesses = dspy.InputField()
    previous_figured_out = dspy.InputField()
    guidelines = dspy.InputField(desc=PLAYER1_GUIDELINES)

    # --- Outputs as two lists ---
    guesses: List[str] = dspy.OutputField(
        desc="A Python list of guess strings, e.g. ['maybe ACTION1 jumps', ...]"
    )
    figured_out: List[str] = dspy.OutputField(
        desc="A Python list of confirmed / figured-out statements."
    )

class Player2(dspy.Signature):
    """ Given the input guesses and figured out items, provide and action.
     Follow the provided guidelines strictly.  Output EXACTLY two lines, no extra text:
    line1: decision type enum (GUESS or INFORMED)
    line2: action enum (RESET|ACTION1|ACTION2|ACTION3|ACTION4|ACTION5|ACTION6|ACTION7)
    Do not include punctuation, explanations, or extra lines."""

    guesses: List[str] = dspy.InputField()
    figured_out: List[str] = dspy.InputField()
    decision_type = dspy.OutputField()
    action = dspy.OutputField()



