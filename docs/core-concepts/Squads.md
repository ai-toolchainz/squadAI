---
title: squadAI Squads
description: Understanding and utilizing squads in the squadAI framework with comprehensive attributes and functionalities.
---

## What is a Squad?
A squad in squadAI represents a collaborative group of agents working together to achieve a set of tasks. Each squad defines the strategy for task execution, agent collaboration, and the overall workflow.

## Squad Attributes

| Attribute                   | Description                                                  |
| :-------------------------- | :----------------------------------------------------------- |
| **Tasks**                   | A list of tasks assigned to the squad.                        |
| **Agents**                  | A list of agents that are part of the squad.                  |
| **Process** *(optional)*    | The process flow (e.g., sequential, hierarchical) the squad follows. |
| **Verbose** *(optional)*    | The verbosity level for logging during execution.            |
| **Manager LLM** *(optional)*| The language model used by the manager agent in a hierarchical process. **Required when using a hierarchical process.** |
| **Function Calling LLM** *(optional)* | If passed, the squad will use this LLM to do function calling for tools for all agents in the squad. Each agent can have its own LLM, which overrides the squad's LLM for function calling. |
| **Config** *(optional)*     | Optional configuration settings for the squad, in `Json` or `Dict[str, Any]` format. |
| **Max RPM** *(optional)*    | Maximum requests per minute the squad adheres to during execution. |
| **Language**  *(optional)*  | Language used for the squad, defaults to English.             |
| **Language File** *(optional)* | Path to the language file to be used for the squad.          |
| **Memory** *(optional)*     | Utilized for storing execution memories (short-term, long-term, entity memory). |
| **Cache** *(optional)*      | Specifies whether to use a cache for storing the results of tools' execution. |
| **Embedder** *(optional)*   | Configuration for the embedder to be used by the squad. mostly used by memory for now       |
| **Full Output** *(optional)*| Whether the squad should return the full output with all tasks outputs or just the final output. |
| **Step Callback** *(optional)* | A function that is called after each step of every agent. This can be used to log the agent's actions or to perform other operations; it won't override the agent-specific `step_callback`. |
| **Task Callback** *(optional)* | A function that is called after the completion of each task. Useful for monitoring or additional operations post-task execution. |
| **Share Squad** *(optional)* | Whether you want to share the complete squad information and execution with the squadAI team to make the library better, and allow us to train models. |
| **Output Log File** *(optional)* | Whether you want to have a file with the complete squad output and execution. You can set it using True and it will default to the folder you are currently and it will be called logs.txt or passing a string with the full path and name of the file. |


!!! note "Squad Max RPM"
    The `max_rpm` attribute sets the maximum number of requests per minute the squad can perform to avoid rate limits and will override individual agents' `max_rpm` settings if you set it.

## Creating a Squad

When assembling a squad, you combine agents with complementary roles and tools, assign tasks, and select a process that dictates their execution order and interaction.

### Example: Assembling a Squad

```python
from squadai import Squad, Agent, Task, Process
from langchain_community.tools import DuckDuckGoSearchRun

# Define agents with specific roles and tools
researcher = Agent(
    role='Senior Research Analyst',
    goal='Discover innovative AI technologies',
    tools=[DuckDuckGoSearchRun()]
)

writer = Agent(
    role='Content Writer',
    goal='Write engaging articles on AI discoveries',
    verbose=True
)

# Create tasks for the agents
research_task = Task(
    description='Identify breakthrough AI technologies',
    agent=researcher
)
write_article_task = Task(
    description='Draft an article on the latest AI technologies',
    agent=writer
)

# Assemble the squad with a sequential process
my_squad = Squad(
    agents=[researcher, writer],
    tasks=[research_task, write_article_task],
    process=Process.sequential,
    full_output=True,
    verbose=True,
)
```

## Memory Utilization

Squads can utilize memory (short-term, long-term, and entity memory) to enhance their execution and learning over time. This feature allows squads to store and recall execution memories, aiding in decision-making and task execution strategies.

## Cache Utilization

Caches can be employed to store the results of tools' execution, making the process more efficient by reducing the need to re-execute identical tasks.

## Squad Usage Metrics

After the squad execution, you can access the `usage_metrics` attribute to view the language model (LLM) usage metrics for all tasks executed by the squad. This provides insights into operational efficiency and areas for improvement.

```python
# Access the squad's usage metrics
squad = Squad(agents=[agent1, agent2], tasks=[task1, task2])
squad.kickoff()
print(squad.usage_metrics)
```

## Squad Execution Process

- **Sequential Process**: Tasks are executed one after another, allowing for a linear flow of work.
- **Hierarchical Process**: A manager agent coordinates the squad, delegating tasks and validating outcomes before proceeding. **Note**: A `manager_llm` is required for this process and it's essential for validating the process flow.

### Kicking Off a Squad

Once your squad is assembled, initiate the workflow with the `kickoff()` method. This starts the execution process according to the defined process flow.

```python
# Start the squad's task execution
result = my_squad.kickoff()
print(result)
```
