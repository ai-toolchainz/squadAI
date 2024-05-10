from squadai.agent import Agent
from squadai.project import agent, task
from squadai.task import Task


class SimpleSquad:
    @agent
    def simple_agent(self):
        return Agent(
            role="Simple Agent", goal="Simple Goal", backstory="Simple Backstory"
        )

    @task
    def simple_task(self):
        return Task(description="Simple Description", expected_output="Simple Output")


def test_agent_memoization():
    squad = SimpleSquad()
    first_call_result = squad.simple_agent()
    second_call_result = squad.simple_agent()

    assert (
        first_call_result is second_call_result
    ), "Agent memoization is not working as expected"


def test_task_memoization():
    squad = SimpleSquad()
    first_call_result = squad.simple_task()
    second_call_result = squad.simple_task()

    assert (
        first_call_result is second_call_result
    ), "Task memoization is not working as expected"
