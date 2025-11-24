
from __future__ import annotations

from enum import Enum
from sqlite3 import Connection
from typing import Any, Dict, List, Optional, Tuple

import inspect
import sys
import types
import json
import logging
import os
import re
import openai
from openai import OpenAI as OpenAIClient
import sqlite3
import textwrap
from typing import ClassVar, List
from PIL import Image
import io

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
    VALID_DT = {"GUESS", "INFORMED"}
    VALID_ACT = {"RESET", "ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5", "ACTION6", "ACTION7"}

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.prev_frame = None
        self.prev_decision_type = 0
        self.prev_action = GameAction.RESET
        self.frame_diff = None
        self.losing_sequences = []
        self.prev_guesses = []
        self.prev_figured_out = []
        self.frame_diff_module = FrameDiffModule()
        self.turn_id = 0 #incremental id of each turn
        self.VALID_DT = {"GUESS", "INFORMED"}
        self.VALID_ACT = {"RESET", "ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5", "ACTION6", "ACTION7"}
        self.agent_db_name = "agent_state.db"
        conn = sqlite3.connect(self.agent_db_name)
        conn.row_factory = sqlite3.Row  # so we can access row["column_name"]

        cur = conn.cursor()
        cur.execute(
            "CREATE TABLE IF NOT EXISTS guesses (id INTEGER PRIMARY KEY, game_id TEXT, card_id TEXT, guess TEXT)")
        cur.execute(
            "CREATE TABLE IF NOT EXISTS figured_outs (id INTEGER PRIMARY KEY, game_id TEXT, card_id TEXT, figs TEXT)")
        cur.execute(
            "CREATE TABLE IF NOT EXISTS losing_actions_seqs (id INTEGER PRIMARY KEY, game_id TEXT, card_id TEXT, losing_seq TEXT)")
        cur.execute(
            "CREATE TABLE IF NOT EXISTS game (card_id TEXT, game_id TEXT, turn_id INT, prev_action TEXT, prev_decision_type TEXT, prev_frame BLOB, frame_diff TEXT)")

    def grid_to_image(self, grid: list[list[list[int]]]) -> Image.Image:
        """Converts a 3D grid of integers into an example PIL image, stacking grid layers horizontally."""
        color_map = [
            (0, 0, 0),
            (0, 0, 170),
            (0, 170, 0),
            (0, 170, 170),
            (170, 0, 0),
            (170, 0, 170),
            (170, 85, 0),
            (170, 170, 170),
            (85, 85, 85),
            (85, 85, 255),
            (85, 255, 85),
            (85, 255, 255),
            (255, 85, 85),
            (255, 85, 255),
            (255, 255, 85),
            (255, 255, 255),
        ]

        height = len(grid[0])
        width = len(grid[0][0])
        num_layers = len(grid)

        # Add a small separator between grids if there are multiple layers
        separator_width = 5 if num_layers > 1 else 0
        total_width = (width * num_layers) + (separator_width * (num_layers - 1))

        image = Image.new("RGB", (total_width, height), "white")
        pixels = image.load()

        for i, grid_layer in enumerate(grid):
            # If you don't need logging, you can ignore inconsistency checks
            if len(grid_layer) != height or len(grid_layer[0]) != width:
                # just skip inconsistent layers
                continue

            offset_x = i * (width + separator_width)
            for y in range(height):
                for x in range(width):
                    color_index = grid_layer[y][x] % 16
                    pixels[x + offset_x, y] = color_map[color_index]

        scale = 10  # 10x bigger pixels
        big_img = image.resize(
        (image.width * scale, image.height * scale),
            resample=Image.NEAREST  # keep the pixel-art look
        )
        return big_img

    def load_prev_state_for_player1(self, game_id: str, card_id: str):
        conn = sqlite3.connect(self.agent_db_name)
        conn.row_factory = sqlite3.Row  # so we can access row["column_name"]
        cur = conn.cursor()

        game_row = cur.execute(
            """
            SELECT *
            FROM game
            WHERE game_id = ?
              AND card_id = ?
            ORDER BY turn_id DESC LIMIT 1
            """,
            (game_id, card_id),
        ).fetchone()

        # if game_row is None:
        #     raise ValueError(f"No game row for game_id={game_id}, card_id={card_id}")

        self.turn_id = game_row["turn_id"]
        prev_frame_bytes = game_row["prev_frame"]
        if prev_frame_bytes:
            self.prev_frame = Image.open(io.BytesIO(prev_frame_bytes))
            self.prev_frame = self.prev_frame.convert("RGBA")
        self.prev_action = game_row["prev_action"]
        self.prev_decision_type = game_row["prev_decision_type"]

        losing_rows = cur.execute(
            """
            SELECT losing_seq
            FROM losing_actions_seqs
            WHERE game_id = ?
                AND card_id = ?
            """,
            (game_id, card_id),
        ).fetchall()
        self.losing_sequences = [json.loads(r["losing_seq"]) for r in losing_rows]

        guesses_rows = cur.execute(
            """
            SELECT guess
            FROM guesses
            WHERE game_id = ?
                AND card_id = ?
            """,
            (game_id, card_id),
        ).fetchall()
        self.prev_guesses = [r["guess"] for r in guesses_rows]

        figout_rows = cur.execute(
            """
            SELECT figs
            FROM figured_outs
            WHERE game_id = ?
                AND card_id = ?
            """,
            (game_id, card_id),
        ).fetchall()
        self.prev_figured_out = [r["figs"] for r in figout_rows]

    def frame_diff_finder(self, current_frame: Image.Image, prev_frame: Image.Image) -> str:
        """
        Uses DSPy + an LLM to describe differences between two game frames.
        Returns:
            A JSON string describing the diff (see FrameDiffSignature).
        """
        # Wrap PIL images as dspy.Image (DSPy will handle encoding/base64 etc.)
        prev_img = dspy.Image(prev_frame)
        current_img = dspy.Image(current_frame)

        prediction = self.frame_diff_module(
            prev_frame=prev_img,
            current_frame=current_img,
        )
        # `prediction.diff_json` is already a string (ideally valid JSON).
        return prediction.diff_json.strip()

    def append_observation(self, card_id, game_id, turn_id,
                        prev_frame_img, frame_diff, guesses, figured_out):
        conn = sqlite3.connect(self.agent_db_name)
        conn.row_factory = sqlite3.Row  # so we can access row["column_name"]
        cur = conn.cursor()

        frame_diff_str = json.dumps(frame_diff)

        # prev_frame as PNG bytes (BLOB)
        buf = io.BytesIO()
        prev_frame_bytes = None
        if prev_frame_img:
            prev_frame_img.save(buf, format="PNG")
            prev_frame_bytes = buf.getvalue()

        cur.execute(
            """
            INSERT INTO game (card_id,
                              game_id,
                              turn_id,
                              prev_frame,
                              frame_diff)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                card_id,
                game_id,
                turn_id,
                prev_frame_bytes,
                frame_diff_str,
            ),
        )

        for guess in guesses:
            cur.execute(
                """
                INSERT INTO guesses (game_id, card_id, guess)
                VALUES (?, ?, ?)
                """,
                (game_id, card_id, guess),
            )

        # Store new figured_out items
        for fig in figured_out:
            cur.execute(
                """
                INSERT INTO figured_outs (game_id, card_id, figs)
                VALUES (?, ?, ?)
                """,
                (game_id, card_id, fig),
            )

        conn.commit()

    def parse_two_line_enums(self, s: str):
        lines = [ln.strip() for ln in s.strip().splitlines() if ln.strip()]
        if len(lines) < 2:
            raise ValueError(f"Expected 2 lines, got {len(lines)}: {lines}")

        dt_raw, act_raw = lines[0], lines[1]
        if dt_raw not in self.VALID_DT:
            raise ValueError(f"Invalid decision type: {dt_raw}")
        if act_raw not in self.VALID_ACT:
            raise ValueError(f"Invalid action: {act_raw}")

        # Map to Python enums
        decision = DecisionType[dt_raw]
        action = GameAction[act_raw]
        return {"decision_type": decision, "action": action, "raw": s}

    def choose_action(
            self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        """Choose which action the Agent should take, fill in any arguments, and return it."""
        action = GameAction.RESET  # default action if LLM doesnt call one

        # -------------------------------------- prepare inputs for observation  ------------------------------------
        current_frame = None
        if latest_frame.frame != []:
            self.load_prev_state_for_player1( self.game_id, self.card_id)
            current_frame = self.grid_to_image(self.frames[-1].frame)
            diff_json_str = self.frame_diff_finder(current_frame, self.prev_frame)
            self.frame_diff = json.loads(diff_json_str)

        logger.info("Sending to Assistant for action...")


        #---------- player1 observes the game and previous step----
        player1 = dspy.Predict(Player1)
        observations = player1(
            current_frame=current_frame,
            prev_frame = self.prev_frame,
            prev_decision_type = self.prev_decision_type,
            prev_action = self.prev_action,
            frame_diff = self.frame_diff,
            losing_sequences = self.losing_sequences,
            prev_guesses = self.prev_guesses,
            prev_figured_out = self.prev_figured_out,
            guidelines = Player1.PLAYER1_GUIDELINES,
        )

        #---------- append player1 output: the observation turn++ ----
        guesses = getattr(observations, "guesses", []) or []
        figured_out = getattr(observations, "figured_out", []) or []
        self.append_observation(
            card_id=self.card_id,
            game_id=self.game_id,
            turn_id=self.turn_id+1,
            prev_frame_img=current_frame,
            frame_diff=self.frame_diff,
            guesses=guesses,
            figured_out=figured_out,
        )


        #---------- call the player 2 ----
        player2 = dspy.Predict(Player2)
        nextAction = player2(
            guesses=guesses,
            figured_out=figured_out,
        )
        try:
            parsed = self.parse_two_line_enums(str(nextAction))
            print("\nPARSED:", parsed["decision_type"], parsed["action"])
        except Exception as e:
            print("Parse error:", e)

        #---------- append player 2 output: action, decision ----


        return action

class FrameDiffSignature(dspy.Signature):
    """
    Given two frames from the SAME game environment:
    - `prev_frame`: the frame BEFORE an action.
    - `current_frame`: the frame AFTER the action.

    Identify **all meaningful visual differences** that relate to gameplay.

    Return a SINGLE JSON object as a string with the following schema:

    {
      "added_objects": [
        {"name": str, "position_hint": str, "color_or_shape": str}
      ],
      "removed_objects": [
        {"name": str, "position_hint": str}
      ],
      "moved_objects": [
        {"name": str, "from": str, "to": str}
      ],
      "ui_changes": [
        {"description": str}
      ],
      "score_or_status_changes": [
        {"description": str}
      ],
      "terminal_event": bool,
      "high_level_summary": str
    }

    - If pre_frame is empty, it's the first frame of the game. no change.
    - Use short, consistent names for objects (e.g. "player", "enemy_1",
      "coin", "projectile", "health_bar", etc.).
    - `position_hint` can be rough ("top-left", "center", "near player").
    - If something is unknown, use null or an empty list, but keep all keys.
    """

    prev_frame: dspy.Image   = dspy.InputField(desc="Frame before the action.")
    current_frame: dspy.Image = dspy.InputField(desc="Frame after the action.")
    diff_json: str           = dspy.OutputField(
        desc="A single JSON object (string) strictly following the schema above."
    )

class FrameDiffModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self._predict = dspy.Predict(FrameDiffSignature)

    def forward(self, prev_frame: dspy.Image, current_frame: dspy.Image):
        """
        prev_frame & current_frame should be dspy.Image objects.
        """
        return self._predict(prev_frame=prev_frame, current_frame=current_frame)

class DecisionType(Enum):
    GUESS : 0
    INFORMED : 1

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
    prev_decision_type = dspy.InputField()
    prev_action = dspy.InputField()
    frame_diff = dspy.InputField()
    losing_sequences = dspy.InputField()
    prev_guesses = dspy.InputField()
    prev_figured_out = dspy.InputField()
    guidelines = dspy.InputField()

    # --- Outputs ---
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

    PLAYER2_GUIDELINES: ClassVar[str] = textwrap.dedent("""
    You're playing a vintage pixel-graphics puzzle along with your friend. You are on the same team. He is Player 1 and you are Player 2. Player 1 sees the game screen; Player 2 (you) performs actions. You two work together as a strong team.

    Each turn, Player 1 receives:
    1. A snapshot of the screen as the current frame.
    2. The previous frame in the same format.
    3. The last move Player 2 (you) has done.
    4. A diff of frames to help identify changed areas.

    To play as a team, you two have come up with a simple tactic.
    Player 1, who sees the game and what each action does, maintains two lists:
    1. A list of "guesses"
    2. A list of "figured_out" things
    Both lists are visible to Player 2 (you).

    You, Player 2, play the game using:
    1. The "guesses" from Player 1
    2. The "figured_out" things from Player 1
    3. Your stage of confidence

    Usually, we have 4 stages of confidence in any game:
    - Stage 1 – We have no guesses and no clue what each action does.
    - Stage 2 – We’ve figured out the actions a bit and know a little about the game environment.
    - Stage 3 – We have many guesses about the game environment and some guesses about how to win.
    - Stage 4 – We’ve figured out the actions, we’ve figured out the environment, and we’ve figured out how to win.

    Most reliably, a game can be won while in Stage 4, but some games can be won in Stages 3, 2, or even 1 by being lucky and guessing the right thing early.

    First, review the lists of "guesses" and "figured_out" things, then estimate your stage of confidence:
    - If there are many guesses and no or only a few "figured_out" items, you are in Stage 1.
    - If you have very few guesses and a high number of "figured_out" items, you are near or in Stage 4, which means you know what each action does and you’ve mostly figured out the game. That’s when it’s time to do the actions that lead to winning.

    Then, based on your stage of confidence, either try a guess by doing an action or make an informed action.

    You must:
    1. Choose a type of decision: GUESS or INFORMED.
    2. Choose one action from: ACTION1, ACTION2, ACTION3, ACTION4, ACTION5, ACTION7, RESET.

    Choose exactly one action. More than one action will be rejected by the game.
    While choosing, trust the current "guesses" list, the current "figured_out" list, and your stage of confidence to choose the best action.
    """).strip()

    # --- Inputs ---
    guesses: List[str] = dspy.InputField()
    figured_out: List[str] = dspy.InputField()

    # --- Outputs ---
    decision_type = dspy.OutputField()
    action = dspy.OutputField()
