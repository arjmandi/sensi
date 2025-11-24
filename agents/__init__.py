from typing import Type, cast

from dotenv import load_dotenv

from .agent import Agent, Playback
from .recorder import Recorder
from .swarm import Swarm

from .templates.llm_agents import (
    LLM,
    SensiLLM,
)
from .templates.random_agent import Random
from .templates.smolagents import SmolCodingAgent, SmolVisionAgent
from .sensi import Sensi


load_dotenv()

AVAILABLE_AGENTS: dict[str, Type[Agent]] = {
    cls.__name__.lower(): cast(Type[Agent], cls)
    for cls in Agent.__subclasses__()
    if cls.__name__ != "Playback"
}

# add all the recording files as valid agent names
for rec in Recorder.list():
    AVAILABLE_AGENTS[rec] = Playback

# update the agent dictionary to include subclasses of LLM class
AVAILABLE_AGENTS["sensillm"] = SensiLLM


__all__ = [
    "Swarm",
    "Random",
    "LLM",
    "SmolCodingAgent",
    "SmolVisionAgent",
    "Agent",
    "Recorder",
    "Playback",
    "Sensi",
    "SensiLLM",
    "AVAILABLE_AGENTS",
]
