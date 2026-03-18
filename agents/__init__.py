from typing import Type, cast

from dotenv import load_dotenv

from .agent import Agent, Playback
from .recorder import Recorder
from .swarm import Swarm
from .sensi_llm import SensiLLM

load_dotenv()

AVAILABLE_AGENTS: dict[str, Type[Agent]] = {
    cls.__name__.lower(): cast(Type[Agent], cls)
    for cls in Agent.__subclasses__()
    if cls.__name__ != "Playback"
}

# add all the recording files as valid agent names
for rec in Recorder.list():
    AVAILABLE_AGENTS[rec] = Playback

AVAILABLE_AGENTS["sensillm"] = SensiLLM

__all__ = [
    "Swarm",
    "Agent",
    "Recorder",
    "Playback",
    "SensiLLM",
    "AVAILABLE_AGENTS",
]
