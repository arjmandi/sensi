import random
import json
import logging
import os
import time
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Optional

import requests
import requests.cookies
from pydantic import ValidationError
from requests import Response
from requests.cookies import RequestsCookieJar

from .recorder import Recorder
from .structs import FrameData, GameAction, GameState, Scorecard
from .tracing import trace_agent_session


from .agent import Agent
from .structs import FrameData, GameAction, GameState
logger = logging.getLogger()

class Sensi(Agent):
    """An agent that always selects actions at random."""

    MAX_ACTIONS = 80

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # seed = int(time.time() * 1000000) + hash(self.game_id) % 1000000
        # random.seed(seed)

    @property
    def name(self) -> str:
        return f"{super().name}.{self.MAX_ACTIONS}"

    # def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
    #     """Decide if the agent is done playing or not."""
    #     return any(
    #         [
    #             latest_frame.state is GameState.WIN,
    #             # uncomment to only let the agent play one time
    #             # latest_frame.state is GameState.GAME_OVER,
    #         ] 
    #     )

    # def choose_action(
    #     self, frames: list[FrameData], latest_frame: FrameData
    # ) -> GameAction:
    #     """Choose which action the Agent should take, fill in any arguments, and return it."""
    #     if latest_frame.state in [GameState.NOT_PLAYED, GameState.GAME_OVER]:
    #         # if game is not started (at init or after GAME_OVER) we need to reset
    #         # add a small delay before resetting after GAME_OVER to avoid timeout
    #         action = GameAction.RESET
    #     else:
    #         # else choose a random action that isnt reset
    #         action = random.choice([a for a in GameAction if a is not GameAction.RESET])
    #
    #     if action.is_simple():
    #         action.reasoning = f"RNG told me to pick {action.value}"
    #     elif action.is_complex():
    #         action.set_data(
    #             {
    #                 "x": random.randint(0, 63),
    #                 "y": random.randint(0, 63),
    #             }
    #         )
    #         action.reasoning = {
    #             "desired_action": f"{action.value}",
    #             "my_reason": "RNG said so!",
    #         }
    #     return action




    
    def main(self) -> None:
        """The main agent loop. Play the game_id until finished, then exits."""
        self.timer = time.time()
        # while (not self.is_done(self.frames, self.frames[-1])):
        #     action = self.choose_action(self.frames, self.frames[-1])
        #     if frame := self.take_action(action):
        #         self.append_frame(frame)
        #         logger.info(
        #             f"{self.game_id} - {action.name}: count {self.action_counter}, score {frame.score}, avg fps {self.fps})"
        #         )
        #     self.action_counter += 1
        
        print("------------iiiiiiiiiiii--------------------------")
        print(self.frames[-1])
        print("------------iiiiiiiiiiii--------------------------")
        
        self.cleanup()

    @property
    def state(self) -> GameState:
        return self.frames[-1].state

    @property
    def score(self) -> int:
        return self.frames[-1].score

    @property
    def seconds(self) -> float:
        return (time.time() - self.timer) * 100 // 1 / 100

    @property
    def fps(self) -> float:
        if self.action_counter == 0:
            return 0.0
        elapsed_time = max(self.seconds, 0.1)
        return round(self.action_counter / elapsed_time, 2)

    @property
    def is_playback(self) -> bool:
        return type(self) is Playback


    # def start_recording(self) -> None:
    #     filename = self.agent_name if self.is_playback else None
    #     self.recorder = Recorder(prefix=self.name, filename=filename)
    #     logger.info(
    #         f"created new recording for {self.name} into {self.recorder.filename}"
    #     )

    def append_frame(self, frame: FrameData) -> None:
        self.frames.append(frame)
        if frame.guid:
            self.guid = frame.guid
        if hasattr(self, "recorder") and not self.is_playback:
            self.recorder.record(json.loads(frame.model_dump_json()))

    # def do_action_request(self, action: GameAction) -> Response:
    #     data = action.action_data.model_dump()
    #     if action == GameAction.RESET:
    #         data["card_id"] = self.card_id
    #     if self.guid:
    #         data["guid"] = self.guid
    #     if action.reasoning:
    #         data["reasoning"] = action.reasoning
    #     if self.game_id:
    #         data["game_id"] = self.game_id
    #
    #     json_str = json.dumps(data)
    #     r = self._session.post(
    #         f"{self.ROOT_URL}/api/cmd/{action.name}",
    #         json=json.loads(json_str),
    #         headers=self.headers,
    #     )
    #     if "error" in r.json():
    #         logger.warning(f"Exception during action request: {r.json()}")
    #     return r

    def take_action(self, action: GameAction) -> Optional[FrameData]:
        """Submits the specific action and gets the next frame."""
        frame_data = self.do_action_request(action).json()
        try:
            frame = FrameData.model_validate(frame_data)
        except ValidationError as e:
            logger.warning(f"Incoming frame data did not validate: {e}")
            return None
        return frame

    def get_scorecard(self) -> Scorecard:
        """Get the scorecard for this agent's game as a Scorecard pydantic object."""
        r = self._session.get(
            f"{self.ROOT_URL}/api/scorecard/{self.card_id}/{self.game_id}",
            timeout=1,
            headers=self.headers,
        )
        response_data = r.json()
        if "error" in response_data:
            logger.warning(f"Exception during scorecard request: {response_data}")
        return Scorecard.model_validate(response_data)

    def cleanup(self, scorecard: Optional[Scorecard] = None) -> None:
        """Called after main loop is finished."""
        if self._cleanup:
            self._cleanup = False  # only cleanup once per agent
            if hasattr(self, "recorder") and not self.is_playback:
                if scorecard:
                    self.recorder.record(scorecard.get(self.game_id))
                else:
                    scorecard_obj = self.get_scorecard()
                    self.recorder.record(scorecard_obj.get(self.game_id))
                logger.info(
                    f"recording for {self.name} is available in {self.recorder.filename}"
                )
            if self.action_counter >= self.MAX_ACTIONS:
                logger.info(
                    f"Exiting: agent reached MAX_ACTIONS of {self.MAX_ACTIONS}, took {self.seconds} seconds ({self.fps} average fps)"
                )
            else:
                logger.info(
                    f"Finishing: agent took {self.action_counter} actions, took {self.seconds} seconds ({self.fps} average fps)"
                )
            if hasattr(self, "_session"):
                self._session.close()

    def is_won(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        """Decide if the agent is done playing or not."""
        return false

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        """Choose which action the Agent should take, fill in any arguments, and return it."""
        return null


class Playback(Agent):

    MAX_ACTIONS = 1000000
    PLAYBACK_FPS = 5

    recorded_actions: list[dict[str, Any]]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.recorder = Recorder(
            prefix=Recorder.get_prefix(self.agent_name),
            guid=Recorder.get_guid(self.agent_name),
        )
        self.recorded_actions = []
        if self.agent_name in Recorder.list():
            try:
                self.recorded_actions = self.filter_actions()
                logger.info(
                    f"Loaded {len(self.recorded_actions)} actions from {self.agent_name}"
                )
            except Exception as e:
                logger.error(f"Failed to load recording {self.agent_name}: {e}")
                self.recorded_actions = []
        else:
            logger.warning(
                f"Recording {self.agent_name} not found in available recordings"
            )

    def filter_actions(self) -> list[dict[str, Any]]:
        return [
            a
            for a in self.recorder.get()
            if "data" in a and "action_input" in a["data"]
        ]

    def is_won(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        return bool(self.action_counter >= len(self.recorded_actions))

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        loop_start_time = time.time()

        if self.action_counter >= len(self.recorded_actions):
            logger.warning(
                f"No more recorded actions available (counter: {self.action_counter}, total: {len(self.recorded_actions)})"
            )
            return GameAction.RESET

        recorded_data = self.recorded_actions[self.action_counter]["data"]
        action_input = recorded_data["action_input"]

        action = GameAction.from_id(action_input["id"])
        data = action_input["data"].copy()
        data["game_id"] = self.game_id
        action.set_data(data)
        if "reasoning" in action_input and action_input["reasoning"] is not None:
            action.reasoning = action_input["reasoning"]

        logger.debug(
            f"Playback action {self.action_counter}: {action.name} with data {data}"
        )

        target_frame_time = 1.0 / getattr(self, "PLAYBACK_FPS", 5)
        elapsed_time = time.time() - loop_start_time
        sleep_time = max(0, target_frame_time - elapsed_time)
        if sleep_time > 0:
            time.sleep(sleep_time)

        return action

    def append_frame(self, frame: FrameData) -> None:
        # overwrite append_frame to not double record
        self.frames.append(frame)
        if frame.guid:
            self.guid = frame.guid

