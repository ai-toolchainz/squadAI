import json
import uuid
from typing import Any, Dict, List, Optional, Union

from langchain_core.callbacks import BaseCallbackHandler
from pydantic import (
    UUID4,
    BaseModel,
    ConfigDict,
    Field,
    InstanceOf,
    Json,
    PrivateAttr,
    field_validator,
    model_validator,
)
from pydantic_core import PydanticCustomError

from squadai.agent import Agent
from squadai.agents.cache import CacheHandler
from squadai.memory.entity.entity_memory import EntityMemory
from squadai.memory.long_term.long_term_memory import LongTermMemory
from squadai.memory.short_term.short_term_memory import ShortTermMemory
from squadai.process import Process
from squadai.task import Task
from squadai.telemetry import Telemetry
from squadai.tools.agent_tools import AgentTools
from squadai.utilities import I18N, FileHandler, Logger, RPMController


class Squad(BaseModel):
    """
    Represents a group of agents, defining how they should collaborate and the tasks they should perform.

    Attributes:
        tasks: List of tasks assigned to the squad.
        agents: List of agents part of this squad.
        manager_llm: The language model that will run manager agent.
        manager_agent: Custom agent that will be used as manager.
        memory: Whether the squad should use memory to store memories of it's execution.
        manager_callbacks: The callback handlers to be executed by the manager agent when hierarchical process is used
        cache: Whether the squad should use a cache to store the results of the tools execution.
        function_calling_llm: The language model that will run the tool calling for all the agents.
        process: The process flow that the squad will follow (e.g., sequential, hierarchical).
        verbose: Indicates the verbosity level for logging during execution.
        config: Configuration settings for the squad.
        max_rpm: Maximum number of requests per minute for the squad execution to be respected.
        prompt_file: Path to the prompt json file to be used for the squad.
        id: A unique identifier for the squad instance.
        full_output: Whether the squad should return the full output with all tasks outputs or just the final output.
        task_callback: Callback to be executed after each task for every agents execution.
        step_callback: Callback to be executed after each step for every agents execution.
        share_squad: Whether you want to share the complete squad infromation and execution with squadAI to make the library better, and allow us to train models.
    """

    __hash__ = object.__hash__  # type: ignore
    _execution_span: Any = PrivateAttr()
    _rpm_controller: RPMController = PrivateAttr()
    _logger: Logger = PrivateAttr()
    _file_handler: FileHandler = PrivateAttr()
    _cache_handler: InstanceOf[CacheHandler] = PrivateAttr(default=CacheHandler())
    _short_term_memory: Optional[InstanceOf[ShortTermMemory]] = PrivateAttr()
    _long_term_memory: Optional[InstanceOf[LongTermMemory]] = PrivateAttr()
    _entity_memory: Optional[InstanceOf[EntityMemory]] = PrivateAttr()

    cache: bool = Field(default=True)
    model_config = ConfigDict(arbitrary_types_allowed=True)
    tasks: List[Task] = Field(default_factory=list)
    agents: List[Agent] = Field(default_factory=list)
    process: Process = Field(default=Process.sequential)
    verbose: Union[int, bool] = Field(default=0)
    memory: bool = Field(
        default=False,
        description="Whether the squad should use memory to store memories of it's execution",
    )
    embedder: Optional[dict] = Field(
        default={"provider": "openai"},
        description="Configuration for the embedder to be used for the squad.",
    )
    usage_metrics: Optional[dict] = Field(
        default=None,
        description="Metrics for the LLM usage during all tasks execution.",
    )
    full_output: Optional[bool] = Field(
        default=False,
        description="Whether the squad should return the full output with all tasks outputs or just the final output.",
    )
    manager_llm: Optional[Any] = Field(
        description="Language model that will run the agent.", default=None
    )
    manager_agent: Optional[Any] = Field(
        description="Custom agent that will be used as manager.", default=None
    )
    manager_callbacks: Optional[List[InstanceOf[BaseCallbackHandler]]] = Field(
        default=None,
        description="A list of callback handlers to be executed by the manager agent when hierarchical process is used",
    )
    function_calling_llm: Optional[Any] = Field(
        description="Language model that will run the agent.", default=None
    )
    config: Optional[Union[Json, Dict[str, Any]]] = Field(default=None)
    id: UUID4 = Field(default_factory=uuid.uuid4, frozen=True)
    share_squad: Optional[bool] = Field(default=False)
    step_callback: Optional[Any] = Field(
        default=None,
        description="Callback to be executed after each step for all agents execution.",
    )
    task_callback: Optional[Any] = Field(
        default=None,
        description="Callback to be executed after each task for all agents execution.",
    )
    max_rpm: Optional[int] = Field(
        default=None,
        description="Maximum number of requests per minute for the squad execution to be respected.",
    )
    prompt_file: str = Field(
        default=None,
        description="Path to the prompt json file to be used for the squad.",
    )
    output_log_file: Optional[Union[bool, str]] = Field(
        default=False,
        description="output_log_file",
    )

    @field_validator("id", mode="before")
    @classmethod
    def _deny_user_set_id(cls, v: Optional[UUID4]) -> None:
        """Prevent manual setting of the 'id' field by users."""
        if v:
            raise PydanticCustomError(
                "may_not_set_field", "The 'id' field cannot be set by the user.", {}
            )

    @field_validator("config", mode="before")
    @classmethod
    def check_config_type(
        cls, v: Union[Json, Dict[str, Any]]
    ) -> Union[Json, Dict[str, Any]]:
        """Validates that the config is a valid type.
        Args:
            v: The config to be validated.
        Returns:
            The config if it is valid.
        """

        # TODO: Improve typing
        return json.loads(v) if isinstance(v, Json) else v  # type: ignore

    @model_validator(mode="after")
    def set_private_attrs(self) -> "Squad":
        """Set private attributes."""
        self._cache_handler = CacheHandler()
        self._logger = Logger(self.verbose)
        if self.output_log_file:
            self._file_handler = FileHandler(self.output_log_file)
        self._rpm_controller = RPMController(max_rpm=self.max_rpm, logger=self._logger)
        self._telemetry = Telemetry()
        self._telemetry.set_tracer()
        self._telemetry.squad_creation(self)
        return self

    @model_validator(mode="after")
    def create_squad_memory(self) -> "Squad":
        """Set private attributes."""
        if self.memory:
            self._long_term_memory = LongTermMemory()
            self._short_term_memory = ShortTermMemory(embedder_config=self.embedder)
            self._entity_memory = EntityMemory(embedder_config=self.embedder)
        return self

    @model_validator(mode="after")
    def check_manager_llm(self):
        """Validates that the language model is set when using hierarchical process."""
        if self.process == Process.hierarchical:
            if not self.manager_llm and not self.manager_agent:
                raise PydanticCustomError(
                    "missing_manager_llm_or_manager_agent",
                    "Attribute `manager_llm` or `manager_agent` is required when using hierarchical process.",
                    {},
                )

            if (self.manager_agent is not None) and (
                self.agents.count(self.manager_agent) > 0
            ):
                raise PydanticCustomError(
                    "manager_agent_in_agents",
                    "Manager agent should not be included in agents list.",
                    {},
                )

        return self

    @model_validator(mode="after")
    def check_config(self):
        """Validates that the squad is properly configured with agents and tasks."""
        if not self.config and not self.tasks and not self.agents:
            raise PydanticCustomError(
                "missing_keys",
                "Either 'agents' and 'tasks' need to be set or 'config'.",
                {},
            )

        if self.config:
            self._setup_from_config()

        if self.agents:
            for agent in self.agents:
                if self.cache:
                    agent.set_cache_handler(self._cache_handler)
                if self.max_rpm:
                    agent.set_rpm_controller(self._rpm_controller)
        return self

    def _setup_from_config(self):
        assert self.config is not None, "Config should not be None."

        """Initializes agents and tasks from the provided config."""
        if not self.config.get("agents") or not self.config.get("tasks"):
            raise PydanticCustomError(
                "missing_keys_in_config", "Config should have 'agents' and 'tasks'.", {}
            )

        self.process = self.config.get("process", self.process)
        self.agents = [Agent(**agent) for agent in self.config["agents"]]
        self.tasks = [self._create_task(task) for task in self.config["tasks"]]

    def _create_task(self, task_config: Dict[str, Any]) -> Task:
        """Creates a task instance from its configuration.

        Args:
            task_config: The configuration of the task.

        Returns:
            A task instance.
        """
        task_agent = next(
            agt for agt in self.agents if agt.role == task_config["agent"]
        )
        del task_config["agent"]
        return Task(**task_config, agent=task_agent)

    def kickoff(self, inputs: Optional[Dict[str, Any]] = {}) -> str:
        """Starts the squad to work on its assigned tasks."""
        self._execution_span = self._telemetry.squad_execution_span(self)
        self._interpolate_inputs(inputs)
        self._set_tasks_callbacks()

        i18n = I18N(prompt_file=self.prompt_file)

        for agent in self.agents:
            agent.i18n = i18n
            agent.squad = self

            if not agent.function_calling_llm:
                agent.function_calling_llm = self.function_calling_llm
            if not agent.step_callback:
                agent.step_callback = self.step_callback

            agent.create_agent_executor()

        metrics = []

        if self.process == Process.sequential:
            result = self._run_sequential_process()
        elif self.process == Process.hierarchical:
            result, manager_metrics = self._run_hierarchical_process()
            metrics.append(manager_metrics)

        else:
            raise NotImplementedError(
                f"The process '{self.process}' is not implemented yet."
            )

        metrics = metrics + [
            agent._token_process.get_summary() for agent in self.agents
        ]
        self.usage_metrics = {
            key: sum([m[key] for m in metrics if m is not None]) for key in metrics[0]
        }

        return result

    def _run_sequential_process(self) -> str:
        """Executes tasks sequentially and returns the final output."""
        task_output = ""
        for task in self.tasks:
            if task.agent.allow_delegation:
                agents_for_delegation = [
                    agent for agent in self.agents if agent != task.agent
                ]
                if len(self.agents) > 1 and len(agents_for_delegation) > 0:
                    task.tools += AgentTools(agents=agents_for_delegation).tools()

            role = task.agent.role if task.agent is not None else "None"
            self._logger.log("debug", f"== Working Agent: {role}", color="bold_purple")
            self._logger.log(
                "info", f"== Starting Task: {task.description}", color="bold_purple"
            )

            if self.output_log_file:
                self._file_handler.log(
                    agent=role, task=task.description, status="started"
                )

            output = task.execute(context=task_output)
            if not task.async_execution:
                task_output = output

            role = task.agent.role if task.agent is not None else "None"
            self._logger.log("debug", f"== [{role}] Task output: {task_output}\n\n")

            if self.output_log_file:
                self._file_handler.log(agent=role, task=task_output, status="completed")

        self._finish_execution(task_output)
        return self._format_output(task_output)

    def _run_hierarchical_process(self) -> str:
        """Creates and assigns a manager agent to make sure the squad completes the tasks."""

        i18n = I18N(prompt_file=self.prompt_file)
        if self.manager_agent is not None:
            self.manager_agent.allow_delegation = True
            manager = self.manager_agent
            if len(manager.tools) > 0:
                raise Exception("Manager agent should not have tools")
            manager.tools = AgentTools(agents=self.agents).tools()
        else:
            manager = Agent(
                role=i18n.retrieve("hierarchical_manager_agent", "role"),
                goal=i18n.retrieve("hierarchical_manager_agent", "goal"),
                backstory=i18n.retrieve("hierarchical_manager_agent", "backstory"),
                tools=AgentTools(agents=self.agents).tools(),
                llm=self.manager_llm,
                verbose=True,
            )

        task_output = ""
        for task in self.tasks:
            self._logger.log("debug", f"Working Agent: {manager.role}")
            self._logger.log("info", f"Starting Task: {task.description}")

            if self.output_log_file:
                self._file_handler.log(
                    agent=manager.role, task=task.description, status="started"
                )

            task_output = task.execute(
                agent=manager, context=task_output, tools=manager.tools
            )

            self._logger.log("debug", f"[{manager.role}] Task output: {task_output}")

            if self.output_log_file:
                self._file_handler.log(
                    agent=manager.role, task=task_output, status="completed"
                )

        self._finish_execution(task_output)
        return self._format_output(task_output), manager._token_process.get_summary()

    def _set_tasks_callbacks(self) -> str:
        """Sets callback for every task suing task_callback"""
        for task in self.tasks:
            if not task.callback:
                task.callback = self.task_callback

    def _interpolate_inputs(self, inputs: Dict[str, Any]) -> str:
        """Interpolates the inputs in the tasks and agents."""
        [task.interpolate_inputs(inputs) for task in self.tasks]
        [agent.interpolate_inputs(inputs) for agent in self.agents]

    def _format_output(self, output: str) -> str:
        """Formats the output of the squad execution."""
        if self.full_output:
            return {
                "final_output": output,
                "tasks_outputs": [task.output for task in self.tasks if task],
            }
        else:
            return output

    def _finish_execution(self, output) -> None:
        if self.max_rpm:
            self._rpm_controller.stop_rpm_counter()
        self._telemetry.end_squad(self, output)

    def __repr__(self):
        return f"Squad(id={self.id}, process={self.process}, number_of_agents={len(self.agents)}, number_of_tasks={len(self.tasks)})"
