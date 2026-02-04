
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import json
import logging
import os
import openai
from openai import OpenAI as OpenAIClient
import sqlite3
import textwrap
from typing import ClassVar, List
from PIL import Image
import io
from dspy.adapters.image_utils import Image as DSPyImage, encode_image
import litellm
litellm.cache = None


from ..agent import Agent
from ..structs import FrameData, GameAction, GameState, Scorecard

# Ensure DSPy/dsp cache stays inside the project workspace so we
# don't rely on a writable home directory (important in sandboxed runs).
os.environ.setdefault("DSP_CACHEDIR", os.path.join(os.getcwd(), "dsp_cache"))
import dspy
def configure_llm(model: str = "openai/gpt-5.2") -> None:
    try:
        lm = dspy.LM(model, cache=False)
        lm.kwargs.pop("max_tokens", None)
        lm.kwargs["max_completion_tokens"] = 4000
        lm.kwargs.setdefault("temperature", 0.3)
        dspy.settings.configure(lm=lm)
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
    score_counter: int

    _latest_tool_call_id: str = "call_12345"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.messages = []
        self.token_counter = 0
        self.game_state = GameState.NOT_PLAYED

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
        return True if self.game_state == GameState.WIN else False


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
        self.current_sequence = []
        self.prev_guesses = []
        self.prev_figured_out = []
        self.frame_diff_module = FrameDiffModule()
        self.turn_id = 0 #incremental id of each turn
        self.game_state = GameState.NOT_PLAYED
        self.VALID_DT = {"GUESS", "INFORMED"}
        self.VALID_ACT = {"RESET", "ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5", "ACTION6", "ACTION7"}
        self.agent_db_name = "agent_state.db"
        conn = sqlite3.connect(self.agent_db_name)
        conn.row_factory = sqlite3.Row  # so we can access row["column_name"]

        cur = conn.cursor()

        cur.execute("""
            CREATE TABLE IF NOT EXISTS guesses (
                game_id TEXT, 
                card_id TEXT, 
                turn_id INT,
                gss TEXT, 
                PRIMARY KEY (game_id, card_id, turn_id)
            )
        """)
        cur.execute(
            "CREATE TABLE IF NOT EXISTS figured_outs (game_id TEXT, card_id TEXT, turn_id INT,figs TEXT, PRIMARY KEY (game_id, card_id, turn_id)) ")
        cur.execute(
            "CREATE TABLE IF NOT EXISTS losing_actions_seqs (game_id TEXT, card_id TEXT, turn_id INT, losing_seq TEXT, PRIMARY KEY (game_id, card_id, turn_id))")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS game (
                card_id TEXT,
                game_id TEXT,
                turn_id INT,
                game_state TEXT,
                prev_action TEXT,
                prev_decision_type TEXT,
                prev_frame BLOB,
                frame_diff TEXT,
                PRIMARY KEY (game_id, card_id, turn_id)
            )
        """)

        # V2: items_to_learn table - stores learning items with state machine and scoring
        cur.execute("""
            CREATE TABLE IF NOT EXISTS items_to_learn (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                game_id TEXT,
                card_id TEXT,
                item_name TEXT NOT NULL,
                state TEXT DEFAULT 'not_reached',
                learning_metric TEXT,
                threshold INTEGER DEFAULT 8,
                current_sense_score INTEGER,
                UNIQUE(game_id, card_id, item_name)
            )
        """)

        # V2: inputs table - key-value store for frame data, game state, etc.
        cur.execute("""
            CREATE TABLE IF NOT EXISTS inputs (
                game_id TEXT,
                card_id TEXT,
                turn_id INTEGER,
                key TEXT,
                value TEXT,
                PRIMARY KEY (game_id, card_id, turn_id, key)
            )
        """)

        conn.commit()
        conn.close()

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
        if big_img.width > 640: #frame size change means we've either lost or won
            self.game_state = GameState.GAME_OVER
            if self.frames[-1].score > self.score_counter:
                # self.game_state = GameState.WIN # we comment this to prevent from breaking playing after a level win
                self.score_counter = self.frames[-1].score

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


        self.turn_id = game_row["turn_id"] + 1
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

        guesses_row = cur.execute(
            """
            SELECT gss
            FROM guesses
            WHERE game_id = ?
                AND card_id = ?
                AND turn_id = ?
            """,
            (game_id, card_id, game_row["turn_id"]),
        ).fetchone()
        self.prev_guesses = json.loads(guesses_row["gss"])


        figout_row = cur.execute(
            """
            SELECT figs
            FROM figured_outs
            WHERE game_id = ?
                AND card_id = ?
                AND turn_id = ?
            """,
            (game_id, card_id, game_row["turn_id"]),
        ).fetchone()
        figured_out = json.loads(figout_row["figs"])
        self.prev_figured_out = figured_out

        conn.close()

    # ==================== V2 Helper Methods ====================

    def initialize_items_to_learn(self, game_id: str, card_id: str) -> None:
        """
        Initialize default learning items for a new game.
        Called once when the first frame is received.
        """
        conn = sqlite3.connect(self.agent_db_name)
        cur = conn.cursor()

        default_items = [
            "learn what each action does in the game",
            "learn how actions affects your energy while playing",
            "learn how to win the game",
        ]

        for item_name in default_items:
            cur.execute(
                """
                INSERT OR IGNORE INTO items_to_learn (game_id, card_id, item_name, state, threshold)
                VALUES (?, ?, ?, 'not_reached', 8)
                """,
                (game_id, card_id, item_name),
            )

        conn.commit()
        conn.close()
        logger.info(f"Initialized {len(default_items)} learning items for game {game_id}")

    def get_current_item_to_learn(self, game_id: str, card_id: str) -> Optional[dict]:
        """
        Returns the current learning item, or None if all items are facts.
        Priority: 1) item with state='learning', 2) first item with state='not_reached'
        """
        conn = sqlite3.connect(self.agent_db_name)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()

        # First, check for an item already in 'learning' state
        learning_row = cur.execute(
            """
            SELECT id, item_name, state, learning_metric, threshold, current_sense_score
            FROM items_to_learn
            WHERE game_id = ? AND card_id = ? AND state = 'learning'
            LIMIT 1
            """,
            (game_id, card_id),
        ).fetchone()

        if learning_row:
            conn.close()
            return dict(learning_row)

        # Otherwise, get the first 'not_reached' item
        not_reached_row = cur.execute(
            """
            SELECT id, item_name, state, learning_metric, threshold, current_sense_score
            FROM items_to_learn
            WHERE game_id = ? AND card_id = ? AND state = 'not_reached'
            ORDER BY id ASC
            LIMIT 1
            """,
            (game_id, card_id),
        ).fetchone()

        conn.close()
        if not_reached_row:
            return dict(not_reached_row)
        return None

    def update_item_state(self, item_id: int, state: str = None,
                          metric: str = None, sense_score: int = None) -> None:
        """Update learning item state, metric, or sense_score."""
        conn = sqlite3.connect(self.agent_db_name)
        cur = conn.cursor()

        updates = []
        values = []
        if state is not None:
            updates.append("state = ?")
            values.append(state)
        if metric is not None:
            updates.append("learning_metric = ?")
            values.append(metric)
        if sense_score is not None:
            updates.append("current_sense_score = ?")
            values.append(sense_score)

        if updates:
            values.append(item_id)
            cur.execute(
                f"UPDATE items_to_learn SET {', '.join(updates)} WHERE id = ?",
                tuple(values),
            )
            conn.commit()
        conn.close()

    def get_facts(self, game_id: str, card_id: str) -> List[str]:
        """Return list of item_names where state='fact'."""
        conn = sqlite3.connect(self.agent_db_name)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()

        rows = cur.execute(
            """
            SELECT item_name
            FROM items_to_learn
            WHERE game_id = ? AND card_id = ? AND state = 'fact'
            ORDER BY id ASC
            """,
            (game_id, card_id),
        ).fetchall()

        conn.close()
        return [row["item_name"] for row in rows]

    def store_input(self, game_id: str, card_id: str, turn_id: int,
                    key: str, value: Any) -> None:
        """Store a key-value input for this turn."""
        conn = sqlite3.connect(self.agent_db_name)
        cur = conn.cursor()

        value_json = json.dumps(value)
        cur.execute(
            """
            INSERT INTO inputs (game_id, card_id, turn_id, key, value)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(game_id, card_id, turn_id, key) DO UPDATE SET
                value = excluded.value
            """,
            (game_id, card_id, turn_id, key, value_json),
        )
        conn.commit()
        conn.close()

    def get_inputs(self, game_id: str, card_id: str, turn_id: int) -> dict:
        """Get all inputs for this turn as a dict."""
        conn = sqlite3.connect(self.agent_db_name)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()

        rows = cur.execute(
            """
            SELECT key, value
            FROM inputs
            WHERE game_id = ? AND card_id = ? AND turn_id = ?
            """,
            (game_id, card_id, turn_id),
        ).fetchall()

        conn.close()
        return {row["key"]: json.loads(row["value"]) for row in rows}

    # ==================== End V2 Helper Methods ====================

    def frame_diff_finder(self, current_frame: Image.Image, prev_frame: Image.Image) -> str:
        """
        Uses DSPy + an LLM to describe differences between two game frames.
        Returns:
            A JSON string describing the diff (see FrameDiffSignature).
        """
        # Wrap PIL images as dspy.Image (DSPy will handle encoding/base64 etc.)
        prev_img = None
        if prev_frame:
            prev_img =  DSPyImage(url=encode_image(prev_frame))
        else:
            prev_img = Image.new("RGBA", (1, 1), (0, 0, 0, 0))
            buf = io.BytesIO()
            prev_img.save(buf, format="PNG")

        current_img = None
        if current_frame:
            current_img =  DSPyImage(url=encode_image(current_frame))
        else:
            current_img = Image.new("RGBA", (1, 1), (0, 0, 0, 0))
            buf = io.BytesIO()
            current_img.save(buf, format="PNG")


        prediction = self.frame_diff_module(
            prev_frame=prev_img,
            current_frame=current_img,
        )
        # `prediction.diff_json` is already a string (ideally valid JSON).
        return prediction.diff_json.strip()

    def append_observation(self, card_id, game_id, turn_id, game_state,
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
                              game_state,
                              prev_frame,
                              frame_diff)
            VALUES (?, ?, ? ,? , ?, ?) ON CONFLICT(card_id, game_id, turn_id) DO
            UPDATE SET 
                game_state = excluded.game_state,
                prev_frame = excluded.prev_frame,
                frame_diff = excluded.frame_diff            
            """,
            (
                card_id,
                game_id,
                turn_id,
                game_state,
                prev_frame_bytes,
                frame_diff_str,
            ),
        )


        gss_json = json.dumps(guesses)
        cur.execute(
            """
            INSERT INTO guesses (game_id, card_id, turn_id, gss)
            VALUES (?, ?, ?,?) ON CONFLICT(card_id, game_id, turn_id) DO
            UPDATE SET
                gss = excluded.gss
            """,
            (game_id, card_id, turn_id, gss_json),
        )


        figs_json = json.dumps(figured_out)
        cur.execute(
            """
            INSERT INTO figured_outs (game_id, card_id, turn_id, figs)
            VALUES (?, ?, ?,?) ON CONFLICT(card_id, game_id, turn_id) DO
            UPDATE SET
                figs = excluded.figs
            """,
            (game_id, card_id, turn_id, figs_json),
        )

        conn.commit()
        conn.close()

    def append_decision(self, card_id, game_id, turn_id,
                        prev_action, prev_decision_type ):
        conn = sqlite3.connect(self.agent_db_name)
        conn.row_factory = sqlite3.Row  # so we can access row["column_name"]
        cur = conn.cursor()

        cur.execute(
            """
            INSERT INTO game (card_id,
                              game_id,
                              turn_id,
                              prev_action,
                              prev_decision_type)
            VALUES (?, ?, ?, ?, ?) ON CONFLICT(card_id, game_id, turn_id) DO
            UPDATE SET
                prev_action = excluded.prev_action,
                prev_decision_type = excluded.prev_decision_type
            """,
            (
                card_id,
                game_id,
                turn_id,
                prev_action,
                prev_decision_type,
            ),
        )

        conn.commit()
        conn.close()

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

        # ==================== 1. FIRST FRAME CHECK (existing) ====================
        current_frame = None
        if latest_frame.frame != []:
            figured_out = []
            self.load_prev_state_for_player1(self.game_id, self.card_id)
            current_frame = self.grid_to_image(self.frames[-1].frame)
            diff_json_str = self.frame_diff_finder(current_frame, self.prev_frame)
            self.frame_diff = json.loads(diff_json_str)
        else:  # start of the play
            figured_out = ["RESET starts the game"]
            guesses = []
            self.score_counter = self.frames[-1].score
            # V2: Initialize default learning items for this game
            self.initialize_items_to_learn(self.game_id, self.card_id)
            self.append_observation(self.card_id, self.game_id, self.turn_id, self.frames[-1].state, current_frame, self.frame_diff, guesses, figured_out)
            self.append_decision(self.card_id, self.game_id, self.turn_id, DecisionType.INFORMED.name, GameAction.RESET.name)
            return action  # the first run, start the game

        logger.info("Sending to Assistant for action...")

        # ==================== 3. STORE INPUTS (NEW) ====================
        # Store key-value inputs for this turn
        self.store_input(self.game_id, self.card_id, self.turn_id, "game_state", self.frames[-1].state.name)
        self.store_input(self.game_id, self.card_id, self.turn_id, "prev_action", self.prev_action)
        self.store_input(self.game_id, self.card_id, self.turn_id, "prev_decision_type", self.prev_decision_type)
        self.store_input(self.game_id, self.card_id, self.turn_id, "frame_diff", self.frame_diff)

        # ==================== 4. LEARNING EVALUATION (NEW) ====================
        # Get the current item to learn
        current_item = self.get_current_item_to_learn(self.game_id, self.card_id)
        current_item_name = ""
        current_sense_score = 0
        sense_reasoning = ""
        facts = self.get_facts(self.game_id, self.card_id)

        if current_item is not None:
            current_item_name = current_item["item_name"]

            # Mark as 'learning' if it was 'not_reached'
            if current_item["state"] == "not_reached":
                self.update_item_state(current_item["id"], state="learning")
                logger.info(f"Started learning item: {current_item_name}")

            # Generate metric if missing
            if current_item["learning_metric"] is None:
                logger.info(f"Generating learning metric for: {current_item_name}")
                metric_gen = dspy.Predict(MetricGeneratorSignature)
                metric_result = metric_gen(item_to_learn=current_item_name)
                learning_metric = getattr(metric_result, "learning_metric", "")
                self.update_item_state(current_item["id"], metric=learning_metric)
                logger.info(f"Generated metric: {learning_metric}")
                # Don't score yet on this turn, proceed to Player1/2
            else:
                # Has metric, evaluate sense score
                logger.info(f"Evaluating sense score for: {current_item_name}")
                inputs_dict = self.get_inputs(self.game_id, self.card_id, self.turn_id)

                sense_scorer = dspy.Predict(SenseScorerSignature)
                score_result = sense_scorer(
                    item_to_learn=current_item_name,
                    learning_metric=current_item["learning_metric"],
                    facts=facts,
                    figured_out=self.prev_figured_out,
                    # inputs=inputs_dict,
                )

                current_sense_score = int(getattr(score_result, "sense_score", 0))
                sense_reasoning = getattr(score_result, "reasoning", "")
                self.update_item_state(current_item["id"], sense_score=current_sense_score)
                # V2: Also store sense score per turn in inputs table for history
                self.store_input(self.game_id, self.card_id, self.turn_id, "sense_score", current_sense_score)
                self.store_input(self.game_id, self.card_id, self.turn_id, "sense_reasoning", sense_reasoning)
                logger.info(f"Sense score for '{current_item_name}': {current_sense_score}/10 - {sense_reasoning}")

                # Check if threshold is met
                threshold = current_item["threshold"] or 8
                if current_sense_score >= threshold:
                    self.update_item_state(current_item["id"], state="fact")
                    logger.info(f"Item '{current_item_name}' marked as FACT (score {current_sense_score} >= threshold {threshold})")
                    # V2: Reset guesses and figured_out for the new learning item
                    self.prev_guesses = []
                    self.prev_figured_out = []
                    logger.info("Reset guesses and figured_out for new learning item")
                    # Get next item to learn
                    facts = self.get_facts(self.game_id, self.card_id)
                    current_item = self.get_current_item_to_learn(self.game_id, self.card_id)
                    current_item_name = current_item["item_name"] if current_item else ""
                    if current_item and current_item["state"] == "not_reached":
                        self.update_item_state(current_item["id"], state="learning")
        else:
            logger.info("All items learned! No current learning target.")

        # ==================== 5. PLAYER 1 - OBSERVER (modified) ====================
        player1 = dspy.Predict(Player1)
        observations = player1(
            current_frame=current_frame,
            prev_frame=self.prev_frame,
            prev_decision_type=self.prev_decision_type,
            prev_action=self.prev_action,
            frame_diff=self.frame_diff,
            losing_sequences=self.losing_sequences,
            prev_guesses=self.prev_guesses,
            prev_figured_out=self.prev_figured_out,
            guidelines=Player1.PLAYER1_GUIDELINES,
            # V2 new inputs
            facts=facts,
            current_item_to_learn=current_item_name,
            current_sense_score=current_sense_score,
            sense_reasoning=sense_reasoning,
        )

        # V2: Append player1 output to previous lists (accumulate until item becomes fact)
        guesses = getattr(observations, "guesses", []) or []
        figured_out = getattr(observations, "figured_out", []) or []

        # Append new items to previous lists, avoiding duplicates
        # guesses = self.prev_guesses + [g for g in new_guesses if g not in self.prev_guesses]
        # figured_out = self.prev_figured_out + [f for f in new_figured_out if f not in self.prev_figured_out]

        self.append_observation(
            card_id=self.card_id,
            game_id=self.game_id,
            turn_id=self.turn_id,
            game_state=self.frames[-1].state.name,
            prev_frame_img=current_frame,
            frame_diff=self.frame_diff,
            guesses=guesses,
            figured_out=figured_out,
        )

        # ==================== 6. PLAYER 2 - ACTOR (modified) ====================
        inputs_dict = self.get_inputs(self.game_id, self.card_id, self.turn_id)

        player2 = dspy.Predict(Player2)
        try:
            nextAction = player2(
                guesses=guesses,
                figured_out=figured_out,
                # V2 new inputs
                facts=facts,
                current_item_to_learn=current_item_name,
                # inputs=inputs_dict,
            )

            parsed = []
            dt = getattr(nextAction, "decision_type", "")
            act = getattr(nextAction, "action", "")
            raw = f"{dt}\n{act}"
            parsed = self.parse_two_line_enums(raw)
            print("\nPARSED:", parsed["decision_type"], parsed["action"])
            self.append_decision(self.card_id, self.game_id, self.turn_id, parsed["decision_type"].name, parsed["action"].name)
        except Exception as e:
            print(f"Player2 LLM error, falling back to previous action: {e}")
            # Fallback to previous action when LLM response is empty/invalid
            prev_act = self.prev_action
            if isinstance(prev_act, str):
                prev_act = GameAction[prev_act]
            prev_dt = self.prev_decision_type
            if isinstance(prev_dt, str):
                prev_dt = DecisionType[prev_dt]
            elif isinstance(prev_dt, int):
                prev_dt = DecisionType(prev_dt)
            parsed = {"decision_type": prev_dt, "action": prev_act}
            print(f"\nFALLBACK: {parsed['decision_type']} {parsed['action']}")
            self.append_decision(self.card_id, self.game_id, self.turn_id, parsed["decision_type"].name, parsed["action"].name)

        # ==================== 7. RETURN ACTION (existing) ====================
        action = parsed["action"]
        self.current_sequence.append(action.name)
        return action

    def cleanup(self, scorecard: Optional[Scorecard] = None) -> None:
        """Called after main loop is finished."""
        if self._cleanup:
            self._cleanup = False  # only cleanup once per agent

            if self.game_state == GameState.GAME_OVER:
                conn = sqlite3.connect(self.agent_db_name)
                conn.row_factory = sqlite3.Row  # so we can access row["column_name"]
                cur = conn.cursor()
                self.losing_sequences.append(self.current_sequence)
                ls_json = json.dumps(self.losing_sequences)
                cur.execute(
                    """
                    INSERT INTO losing_actions_seqs (game_id, card_id, turn_id, losing_seq)
                    VALUES (?, ?, ?, ?) ON CONFLICT(card_id, game_id, turn_id) DO
                    UPDATE SET
                        losing_seq = excluded.losing_seq
                    """,
                    (self.game_id, self.card_id, self.turn_id, ls_json),
                )
                conn.commit()

            if hasattr(self, "recorder") and not self.is_playback:
                if scorecard:
                    self.recorder.record(scorecard.get(self.game_id))
                else:
                    scorecard_obj = self.get_scorecard()
                    self.recorder.record(scorecard_obj.get(self.game_id))
                logger.info(
                    f"recording for {self.name} is available in {self.recorder.filename}"
                )

            logger.info(
                    f"Finishing: agent took {self.action_counter} actions, took {self.seconds} seconds ({self.fps} average fps)"
                )
            if hasattr(self, "_session"):
                self._session.close()

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


# ==================== V2 DSPy Signatures ====================

class MetricGeneratorSignature(dspy.Signature):
    """
    Given an item the agent needs to learn about a game environment,
    generate a metric to verify that the item has been learned.

    Think about:
    - What observations would confirm understanding?
    - What test actions could validate the learning?
    - What patterns in game feedback would indicate mastery?

    """

    item_to_learn: str = dspy.InputField(
        desc="The item/concept the agent needs to learn"
    )

    learning_metric: str = dspy.OutputField(
        desc="A clear description of how to verify that this item has been learned. "
             "What observations or outcomes would confirm understanding? Desired output is a criteria and a description that a judge will use when to give a score on how good a grasp learner has on the item. This metric will later on will be used to give a score to the learning. the score will be between 1 to 10. 1 lower understanding of the item to learn, and 10 highest score of understanding."
    )


class SenseScorerSignature(dspy.Signature):
    """
    Score the agent's understanding of a learning item from 1 to 10.

    Scoring guide:
    - 1-3: No/minimal understanding
    - 4-6: Partial understanding, still exploring
    - 7-8: Good understanding, some gaps
    - 9-10: Complete understanding, can apply reliably
    """

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
    # inputs: dict = dspy.InputField(
    #     desc="Current game inputs (frame_summary, game_state, prev_action, etc.)"
    # )

    sense_score: int = dspy.OutputField(
        desc="Score from 1-10 indicating learning progress."
    )
    reasoning: str = dspy.OutputField(
        desc="Brief explanation for the score"
    )


# ==================== End V2 DSPy Signatures ====================


class DecisionType(Enum):
    GUESS = 0
    INFORMED = 1

class Player1(dspy.Signature):
    """Given the inputs, Return two lists: guesses and figured_out.
    Follow the provided guidelines strictly. Output only valid Python lists (no extra text)."""

    PLAYER1_GUIDELINES: ClassVar[str] = textwrap.dedent("""
    You're playing a vintage pixel-graphics puzzle along with your friend. You are on the same team. You're Player 1 and he is Player 2. Player 1 sees the game screen, Player 2 performs actions. You two work like a team very well.

    To play as a team you two have come up with a simple tactic. Player 1 (you), who sees the game and what each action does, maintains two lists:
    1) a list of "guesses"
    2) a list of "figured_out" things.
    Both lists will be visible to Player 2.

    Player 1 (you), in each turn receives:
    1. A snapshot of the screen as the current frame: a photo
    2. Previous frame in the same format
    3. Previous type of decision Player 2 has done: GUESS or INFORMED
    4. Previous action Player 2 has done: ACTION1, ACTION2, ACTION3, ACTION4, ACTION5, ACTION7, RESET
    5. Diff of frames to help identify changed areas
    8. Current item to learn: your focus and most important objective. 
    9. Sense score current_sense_score: a score that a judge has given you based on how much you have figured out about the item to learn
    10. Score reason sense_reasoning: the reason of your score
    11. Facts: items your team has definitively learned and confirmed in previous runs
    

    You, Player 1, populate the "guesses" list and "figured_out" list as below:
    1. Develop guesses about the items to learn by asking: "If this action made this change, then this action is what?" and write it in the "guesses" list.
       Consider:
         1) the last action
         2) the changes in the frames
         3) previous guesses
         4) previous figured_out things
         5) previous sequences that led to game over
         6) the current item to learn (focus your guesses toward understanding this)
       Then write simple guesses.
    2. Write all guesses you can make. For example, if you guess an area of the screen is showing a character, if there's a point counter, if there's a timer, etc. Be creative about guesses.
    3. If, based on the last action, you have figured out what each action does, move guesses about that to the "figured_out" list. Write them as simple actions. For example, "ACTION1 jumps over things".
    4. Read the previous action from the inputs, and the difference it made in the frame "frame diff" this is most important piece of information to update your "figured_out" list. aggressively try to figure out the item to learn and check what the sense score jusde is asking for.
    5. If, based on the last action, you have figured out things about the game environment or how to win, put them in "figured_out". For example, "taking the character to the door makes us win".
    6. If you want your friend, Player 2, to further try an action to see if that makes you progress or lose, add it to the "guesses" list. When your friend is not certain which move to pick, he tries more guesses.
    7. If, based on current figured_out items, you have further guesses, add them to your "guesses" list for next rounds.
    8. Remove guesses that seem unlikely at this point. You have to forget useless guesses; otherwise everything will be a guess and the list becomes useless.
    9. Develop and curate your guesses and figured out focused on the item to learn
    10. Review the "figured_out" list and if things contradict each other, make a decision and provide a sane list to Player 2.

    You can only communicate with Player 2 through guesses list and the figured out list. The goal is to learn the item and get a good sense score. Player 2 tries to resolve your doubt (Player 1 doubt) about guesses by choosing actions that best help to move guesses to figured out about the item to learn.

    So help him with smart "guesses" and certain "figured_out" things. Be patient with the list. Player 2 only has one action at a time but you can play as many times as you want. You play action by action to figure out the game and then win it.
            
    LIST MANAGEMENT RULES for figured out items and guesses:
    When outputting guesses and figured_out, you are EDITING the previous lists:
    - KEEP: Items still valid based on current observations
    - EDIT: Items that need wording updates (e.g., more specific after new info)
    - REMOVE: Items contradicted by evidence or no longer relevant
    - ADD: New items based on latest frame/action
    - PROMOTE: Move confirmed guesses from guesses → figured_out
    
    Do NOT start fresh each turn. Curate the existing lists.

    IMPORTANT: You have access to "facts" - these are items your team has definitively learned and confirmed through the learning system.
    Use the facts to inform your analysis, but focus your guesses and observations toward understanding the current item to learn.
    Your goal is to learn the item to learn by steering the Player 2 through maintaining the guesses and figured out lists
    
    SENSE SCORE FEEDBACK:
    You receive a current_sense_score (1-10) and sense_reasoning explaining why the score was given.
    - Score 1-3: Very low understanding. Focus on basic exploration and generating diverse guesses.
    - Score 4-6: Partial understanding. Refine your guesses based on the reasoning feedback.
    - Score 7-8: Good understanding with gaps. The reasoning tells you what's still missing - target those gaps.
    - Score 9-10: Near mastery. Consolidate figured_out items and prepare to move to the next learning item.

    Use the sense_reasoning to understand WHY the score is what it is. If the reasoning says "missing understanding of X",
    focus your guesses and observations on X. This feedback loop helps you steer the game toward learning efficiently.
    """).strip()

    # --- Inputs ---
    current_frame = dspy.InputField()
    prev_frame = dspy.InputField()
    prev_decision_type = dspy.InputField()
    prev_action = dspy.InputField()
    frame_diff = dspy.InputField()
    losing_sequences = dspy.InputField()
    prev_guesses = dspy.InputField(
        desc="Current guesses list to curate: keep valid, edit if needed, remove outdated"
    )
    prev_figured_out = dspy.InputField(
        desc="Current figured_out list to curate: keep confirmed, edit if needed, remove if contradicted"
    )
    guidelines = dspy.InputField()

    # --- V2 New Inputs ---
    facts: List[str] = dspy.InputField(
        desc="Confirmed facts from items marked as learned through the learning system"
    )
    current_item_to_learn: str = dspy.InputField(
        desc="The specific item currently being learned - focus guesses toward this"
    )
    current_sense_score: int = dspy.InputField(
        desc="Current sense score (1-10) for the learning item. Higher means closer to mastery."
    )
    sense_reasoning: str = dspy.InputField(
        desc="Explanation from the sense scorer about why the current score was given and what's missing."
    )

    # --- Outputs ---
    guesses: List[str] = dspy.OutputField(
        desc="A Python list of guess strings, Updated guesses list: preserved items + edits + new guesses. Remove unlikely ones."
    )
    figured_out: List[str] = dspy.OutputField(
        desc="A Python list of figured-out statements. Updated figured_out list: preserved confirmations + promoted guesses + new discoveries."
    )

class Player2(dspy.Signature):
    """ Given the input guesses and figured out items, provide an action to make sense of the current item to learn.
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
    3. The "facts" - confirmed learnings from items your team has definitively learned
    4. The current item to learn - your current learning target

    First, review the lists of "guesses" and "figured_out" things, and then decide how you can help the Player 1 to learn the item:
    Try to resolve Player 1 doubt about guesses by choosing actions that best help to move guesses to figured out about the item to learn.

    You must:
    1. Choose a type of decision: GUESS or INFORMED.
    2. Choose one action from: ACTION1, ACTION2, ACTION3, ACTION4, ACTION5, ACTION7, RESET.

    Choose exactly one action. More than one action will be rejected.
    """).strip()

    # --- Inputs ---
    guesses: List[str] = dspy.InputField()
    figured_out: List[str] = dspy.InputField()

    # --- V2 New Inputs ---
    facts: List[str] = dspy.InputField(
        desc="Confirmed facts from items marked as learned through the learning system"
    )
    current_item_to_learn: str = dspy.InputField(
        desc="The specific item currently being learned - choose actions to make sense of this"
    )
    # inputs: dict = dspy.InputField(
    #     desc="Current game inputs (frame data, game state, prev_action, etc.)"
    # )

    # --- Outputs ---
    decision_type = dspy.OutputField()
    action = dspy.OutputField()
