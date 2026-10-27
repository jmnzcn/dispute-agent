import pytest

from dispute_agent.world import World


@pytest.fixture(scope="session")
def world() -> World:
    return World.load()
