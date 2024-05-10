import asyncio
import json
import os
import platform
from typing import Any

import pkg_resources
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import Status, StatusCode


class Telemetry:
    """A class to handle anonymous telemetry for the squadai package.

    The data being collected is for development purpose, all data is anonymous.

    There is NO data being collected on the prompts, tasks descriptions
    agents backstories or goals nor responses or any data that is being
    processed by the agents, nor any secrets and env vars.

    Data collected includes:
    - Version of squadAI
    - Version of Python
    - General OS (e.g. number of CPUs, macOS/Windows/Linux)
    - Number of agents and tasks in a squad
    - Squad Process being used
    - If Agents are using memory or allowing delegation
    - If Tasks are being executed in parallel or sequentially
    - Language model being used
    - Roles of agents in a squad
    - Tools names available

    Users can opt-in to sharing more complete data suing the `share_squad`
    attribute in the Squad class.
    """

    def __init__(self):
        self.ready = False
        self.trace_set = False
        try:
            telemetry_endpoint = "https://telemetry.squadai.com:4319"
            self.resource = Resource(
                attributes={SERVICE_NAME: "squadAI-telemetry"},
            )
            self.provider = TracerProvider(resource=self.resource)

            processor = BatchSpanProcessor(
                OTLPSpanExporter(
                    endpoint=f"{telemetry_endpoint}/v1/traces",
                    timeout=30,
                )
            )

            self.provider.add_span_processor(processor)
            self.ready = True
        except BaseException as e:
            if isinstance(
                e,
                (SystemExit, KeyboardInterrupt, GeneratorExit, asyncio.CancelledError),
            ):
                raise  # Re-raise the exception to not interfere with system signals
            self.ready = False

    def set_tracer(self):
        if self.ready and not self.trace_set:
            try:
                trace.set_tracer_provider(self.provider)
                self.trace_set = True
            except Exception:
                self.ready = False
                self.trace_set = False

    def squad_creation(self, squad):
        """Records the creation of a squad."""
        if self.ready:
            try:
                tracer = trace.get_tracer("squadai.telemetry")
                span = tracer.start_span("Squad Created")
                self._add_attribute(
                    span,
                    "squadai_version",
                    pkg_resources.get_distribution("squadai").version,
                )
                self._add_attribute(span, "python_version", platform.python_version())
                self._add_attribute(span, "squad_id", str(squad.id))
                self._add_attribute(span, "squad_process", squad.process)
                self._add_attribute(
                    span, "squad_language", squad.prompt_file if squad.i18n else "None"
                )
                self._add_attribute(span, "squad_memory", squad.memory)
                self._add_attribute(span, "squad_number_of_tasks", len(squad.tasks))
                self._add_attribute(span, "squad_number_of_agents", len(squad.agents))
                self._add_attribute(
                    span,
                    "squad_agents",
                    json.dumps(
                        [
                            {
                                "id": str(agent.id),
                                "role": agent.role,
                                "verbose?": agent.verbose,
                                "max_iter": agent.max_iter,
                                "max_rpm": agent.max_rpm,
                                "i18n": agent.i18n.prompt_file,
                                "llm": json.dumps(self._safe_llm_attributes(agent.llm)),
                                "delegation_enabled?": agent.allow_delegation,
                                "tools_names": [
                                    tool.name.casefold() for tool in agent.tools
                                ],
                            }
                            for agent in squad.agents
                        ]
                    ),
                )
                self._add_attribute(
                    span,
                    "squad_tasks",
                    json.dumps(
                        [
                            {
                                "id": str(task.id),
                                "async_execution?": task.async_execution,
                                "agent_role": task.agent.role if task.agent else "None",
                                "tools_names": [
                                    tool.name.casefold() for tool in task.tools
                                ],
                            }
                            for task in squad.tasks
                        ]
                    ),
                )
                self._add_attribute(span, "platform", platform.platform())
                self._add_attribute(span, "platform_release", platform.release())
                self._add_attribute(span, "platform_system", platform.system())
                self._add_attribute(span, "platform_version", platform.version())
                self._add_attribute(span, "cpus", os.cpu_count())
                span.set_status(Status(StatusCode.OK))
                span.end()
            except Exception:
                pass

    def tool_repeated_usage(self, llm: Any, tool_name: str, attempts: int):
        """Records the repeated usage 'error' of a tool by an agent."""
        if self.ready:
            try:
                tracer = trace.get_tracer("squadai.telemetry")
                span = tracer.start_span("Tool Repeated Usage")
                self._add_attribute(
                    span,
                    "squadai_version",
                    pkg_resources.get_distribution("squadai").version,
                )
                self._add_attribute(span, "tool_name", tool_name)
                self._add_attribute(span, "attempts", attempts)
                if llm:
                    self._add_attribute(
                        span, "llm", json.dumps(self._safe_llm_attributes(llm))
                    )
                span.set_status(Status(StatusCode.OK))
                span.end()
            except Exception:
                pass

    def tool_usage(self, llm: Any, tool_name: str, attempts: int):
        """Records the usage of a tool by an agent."""
        if self.ready:
            try:
                tracer = trace.get_tracer("squadai.telemetry")
                span = tracer.start_span("Tool Usage")
                self._add_attribute(
                    span,
                    "squadai_version",
                    pkg_resources.get_distribution("squadai").version,
                )
                self._add_attribute(span, "tool_name", tool_name)
                self._add_attribute(span, "attempts", attempts)
                if llm:
                    self._add_attribute(
                        span, "llm", json.dumps(self._safe_llm_attributes(llm))
                    )
                span.set_status(Status(StatusCode.OK))
                span.end()
            except Exception:
                pass

    def tool_usage_error(self, llm: Any):
        """Records the usage of a tool by an agent."""
        if self.ready:
            try:
                tracer = trace.get_tracer("squadai.telemetry")
                span = tracer.start_span("Tool Usage Error")
                self._add_attribute(
                    span,
                    "squadai_version",
                    pkg_resources.get_distribution("squadai").version,
                )
                if llm:
                    self._add_attribute(
                        span, "llm", json.dumps(self._safe_llm_attributes(llm))
                    )
                span.set_status(Status(StatusCode.OK))
                span.end()
            except Exception:
                pass

    def squad_execution_span(self, squad):
        """Records the complete execution of a squad.
        This is only collected if the user has opted-in to share the squad.
        """
        if (self.ready) and (squad.share_squad):
            try:
                tracer = trace.get_tracer("squadai.telemetry")
                span = tracer.start_span("Squad Execution")
                self._add_attribute(
                    span,
                    "squadai_version",
                    pkg_resources.get_distribution("squadai").version,
                )
                self._add_attribute(span, "squad_id", str(squad.id))
                self._add_attribute(
                    span,
                    "squad_agents",
                    json.dumps(
                        [
                            {
                                "id": str(agent.id),
                                "role": agent.role,
                                "goal": agent.goal,
                                "backstory": agent.backstory,
                                "verbose?": agent.verbose,
                                "max_iter": agent.max_iter,
                                "max_rpm": agent.max_rpm,
                                "i18n": agent.i18n.prompt_file,
                                "llm": json.dumps(self._safe_llm_attributes(agent.llm)),
                                "delegation_enabled?": agent.allow_delegation,
                                "tools_names": [
                                    tool.name.casefold() for tool in agent.tools
                                ],
                            }
                            for agent in squad.agents
                        ]
                    ),
                )
                self._add_attribute(
                    span,
                    "squad_tasks",
                    json.dumps(
                        [
                            {
                                "id": str(task.id),
                                "description": task.description,
                                "async_execution?": task.async_execution,
                                "output": task.expected_output,
                                "agent_role": task.agent.role if task.agent else "None",
                                "context": (
                                    [task.description for task in task.context]
                                    if task.context
                                    else "None"
                                ),
                                "tools_names": [
                                    tool.name.casefold() for tool in task.tools
                                ],
                            }
                            for task in squad.tasks
                        ]
                    ),
                )
                return span
            except Exception:
                pass

    def end_squad(self, squad, output):
        if (self.ready) and (squad.share_squad):
            try:
                self._add_attribute(
                    squad._execution_span,
                    "squadai_version",
                    pkg_resources.get_distribution("squadai").version,
                )
                self._add_attribute(squad._execution_span, "squad_output", output)
                self._add_attribute(
                    squad._execution_span,
                    "squad_tasks_output",
                    json.dumps(
                        [
                            {
                                "id": str(task.id),
                                "description": task.description,
                                "output": task.output.raw_output,
                            }
                            for task in squad.tasks
                        ]
                    ),
                )
                squad._execution_span.set_status(Status(StatusCode.OK))
                squad._execution_span.end()
            except Exception:
                pass

    def _add_attribute(self, span, key, value):
        """Add an attribute to a span."""
        try:
            return span.set_attribute(key, value)
        except Exception:
            pass

    def _safe_llm_attributes(self, llm):
        attributes = ["name", "model_name", "base_url", "model", "top_k", "temperature"]
        if llm:
            safe_attributes = {k: v for k, v in vars(llm).items() if k in attributes}
            safe_attributes["class"] = llm.__class__.__name__
            return safe_attributes
        return {}
