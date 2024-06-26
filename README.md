<div align="center">

![Logo of squadAI, two people rowing on a boat](./docs/squadai_logo.svg)

# **squadAI**

🤖 **squadAI**: Cutting-edge framework for orchestrating role-playing, autonomous AI agents. By fostering collaborative intelligence, SquadAI empowers agents to work together seamlessly, tackling complex tasks.

<h3>

[Homepage](https://www.squadai.io/) | [Documentation](https://docs.squadai.com/) | [Chat with Docs](https://chatg.pt/DWjSBZn) | [Examples](https://github.com/joaomdmoura/squadai-examples) | [Discord](https://discord.com/invite/X4JWnZnxPb)

</h3>

[![GitHub Repo stars](https://img.shields.io/github/stars/joaomdmoura/squadAI)](https://github.com/joaomdmoura/squadAI)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

</div>

## Table of contents

- [Why SquadAI?](#why-squadai)
- [Getting Started](#getting-started)
- [Key Features](#key-features)
- [Examples](#examples)
  - [Quick Tutorial](#quick-tutorial)
  - [Write Job Descriptions](#write-job-descriptions)
  - [Trip Planner](#trip-planner)
  - [Stock Analysis](#stock-analysis)
- [Connecting Your Squad to a Model](#connecting-your-squad-to-a-model)
- [How SquadAI Compares](#how-squadai-compares)
- [Contribution](#contribution)
- [Telemetry](#telemetry)
- [License](#license)

## Why SquadAI?

The power of AI collaboration has too much to offer.
SquadAI is designed to enable AI agents to assume roles, share goals, and operate in a cohesive unit - much like a well-oiled squad. Whether you're building a smart assistant platform, an automated customer service ensemble, or a multi-agent research team, SquadAI provides the backbone for sophisticated multi-agent interactions.

## Getting Started

To get started with SquadAI, follow these simple steps:

### 1. Installation

```shell
pip install squadai
```

If you want to install the 'squadai' package along with its optional features that include additional tools for agents, you can do so by using the following command: pip install 'squadai[tools]'. This command installs the basic package and also adds extra components which require more dependencies to function."

```shell
pip install 'squadai[tools]'
```

### 2. Setting Up Your Squad

```python
import os
from squadai import Agent, Task, Squad, Process
from crewai_tools import SerperDevTool

os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY"
os.environ["SERPER_API_KEY"] = "Your Key" # serper.dev API key

# You can choose to use a local model through Ollama for example. See https://docs.squadai.com/how-to/LLM-Connections/ for more information.

# os.environ["OPENAI_API_BASE"] = 'http://localhost:11434/v1'
# os.environ["OPENAI_MODEL_NAME"] ='openhermes'  # Adjust based on available model
# os.environ["OPENAI_API_KEY"] ='sk-111111111111111111111111111111111111111111111111'

search_tool = SerperDevTool()

# Define your agents with roles and goals
researcher = Agent(
  role='Senior Research Analyst',
  goal='Uncover cutting-edge developments in AI and data science',
  backstory="""You work at a leading tech think tank.
  Your expertise lies in identifying emerging trends.
  You have a knack for dissecting complex data and presenting actionable insights.""",
  verbose=True,
  allow_delegation=False,
  tools=[search_tool]
  # You can pass an optional llm attribute specifying what model you wanna use.
  # It can be a local model through Ollama / LM Studio or a remote
  # model like OpenAI, Mistral, Antrophic or others (https://docs.squadai.com/how-to/LLM-Connections/)
  #
  # import os
  # os.environ['OPENAI_MODEL_NAME'] = 'gpt-3.5-turbo'
  #
  # OR
  #
  # from langchain_openai import ChatOpenAI
  # llm=ChatOpenAI(model_name="gpt-3.5", temperature=0.7)
)
writer = Agent(
  role='Tech Content Strategist',
  goal='Craft compelling content on tech advancements',
  backstory="""You are a renowned Content Strategist, known for your insightful and engaging articles.
  You transform complex concepts into compelling narratives.""",
  verbose=True,
  allow_delegation=True
)

# Create tasks for your agents
task1 = Task(
  description="""Conduct a comprehensive analysis of the latest advancements in AI in 2024.
  Identify key trends, breakthrough technologies, and potential industry impacts.""",
  expected_output="Full analysis report in bullet points",
  agent=researcher
)

task2 = Task(
  description="""Using the insights provided, develop an engaging blog
  post that highlights the most significant AI advancements.
  Your post should be informative yet accessible, catering to a tech-savvy audience.
  Make it sound cool, avoid complex words so it doesn't sound like AI.""",
  expected_output="Full blog post of at least 4 paragraphs",
  agent=writer
)

# Instantiate your squad with a sequential process
squad = Squad(
  agents=[researcher, writer],
  tasks=[task1, task2],
  verbose=2, # You can set it to 1 or 2 to different logging levels
)

# Get your squad to work!
result = squad.kickoff()

print("######################")
print(result)
```

In addition to the sequential process, you can use the hierarchical process, which automatically assigns a manager to the defined squad to properly coordinate the planning and execution of tasks through delegation and validation of results. [See more about the processes here](https://docs.squadai.com/core-concepts/Processes/).

## Key Features

- **Role-Based Agent Design**: Customize agents with specific roles, goals, and tools.
- **Autonomous Inter-Agent Delegation**: Agents can autonomously delegate tasks and inquire amongst themselves, enhancing problem-solving efficiency.
- **Flexible Task Management**: Define tasks with customizable tools and assign them to agents dynamically.
- **Processes Driven**: Currently only supports `sequential` task execution and `hierarchical` processes, but more complex processes like consensual and autonomous are being worked on.
- **Save output as file**: Save the output of individual tasks as a file, so you can use it later.
- **Parse output as Pydantic or Json**: Parse the output of individual tasks as a Pydantic model or as a Json if you want to.
- **Works with Open Source Models**: Run your squad using Open AI or open source models refer to the [Connect squadAI to LLMs](https://docs.squadai.com/how-to/LLM-Connections/) page for details on configuring your agents' connections to models, even ones running locally!

![SquadAI Mind Map](./docs/squadAI-mindmap.png "SquadAI Mind Map")

## Examples

You can test different real life examples of AI squads in the [squadAI-examples repo](https://github.com/joaomdmoura/squadAI-examples?tab=readme-ov-file):

- [Landing Page Generator](https://github.com/joaomdmoura/squadAI-examples/tree/main/landing_page_generator)
- [Having Human input on the execution](https://docs.squadai.com/how-to/Human-Input-on-Execution)
- [Trip Planner](https://github.com/joaomdmoura/squadAI-examples/tree/main/trip_planner)
- [Stock Analysis](https://github.com/joaomdmoura/squadAI-examples/tree/main/stock_analysis)

### Quick Tutorial

[![SquadAI Tutorial](https://img.youtube.com/vi/tnejrr-0a94/maxresdefault.jpg)](https://www.youtube.com/watch?v=tnejrr-0a94 "SquadAI Tutorial")

### Write Job Descriptions

[Check out code for this example](https://github.com/joaomdmoura/squadAI-examples/tree/main/job-posting) or watch a video below:

[![Jobs postings](https://img.youtube.com/vi/u98wEMz-9to/maxresdefault.jpg)](https://www.youtube.com/watch?v=u98wEMz-9to "Jobs postings")

### Trip Planner

[Check out code for this example](https://github.com/joaomdmoura/squadAI-examples/tree/main/trip_planner) or watch a video below:

[![Trip Planner](https://img.youtube.com/vi/xis7rWp-hjs/maxresdefault.jpg)](https://www.youtube.com/watch?v=xis7rWp-hjs "Trip Planner")

### Stock Analysis

[Check out code for this example](https://github.com/joaomdmoura/squadAI-examples/tree/main/stock_analysis) or watch a video below:

[![Stock Analysis](https://img.youtube.com/vi/e0Uj4yWdaAg/maxresdefault.jpg)](https://www.youtube.com/watch?v=e0Uj4yWdaAg "Stock Analysis")

## Connecting Your Squad to a Model

squadAI supports using various LLMs through a variety of connection options. By default your agents will use the OpenAI API when querying the model. However, there are several other ways to allow your agents to connect to models. For example, you can configure your agents to use a local model via the Ollama tool.

Please refer to the [Connect squadAI to LLMs](https://docs.squadai.com/how-to/LLM-Connections/) page for details on configuring you agents' connections to models.

## How SquadAI Compares

- **Autogen**: While Autogen does good in creating conversational agents capable of working together, it lacks an inherent concept of process. In Autogen, orchestrating agents' interactions requires additional programming, which can become complex and cumbersome as the scale of tasks grows.

- **ChatDev**: ChatDev introduced the idea of processes into the realm of AI agents, but its implementation is quite rigid. Customizations in ChatDev are limited and not geared towards production environments, which can hinder scalability and flexibility in real-world applications.

**SquadAI's Advantage**: SquadAI is built with production in mind. It offers the flexibility of Autogen's conversational agents and the structured process approach of ChatDev, but without the rigidity. SquadAI's processes are designed to be dynamic and adaptable, fitting seamlessly into both development and production workflows.

## Contribution

SquadAI is open-source and we welcome contributions. If you're looking to contribute, please:

- Fork the repository.
- Create a new branch for your feature.
- Add your feature or improvement.
- Send a pull request.
- We appreciate your input!

### Installing Dependencies

```bash
poetry lock
poetry install
```

### Virtual Env

```bash
poetry shell
```

### Pre-commit hooks

```bash
pre-commit install
```

### Running Tests

```bash
poetry run pytest
```

### Running static type checks

```bash
poetry run pyright
```

### Packaging

```bash
poetry build
```

### Installing Locally

```bash
pip install dist/*.tar.gz
```

## Telemetry

SquadAI uses anonymous telemetry to collect usage data with the main purpose of helping us improve the library by focusing our efforts on the most used features, integrations and tools.

There is NO data being collected on the prompts, tasks descriptions agents backstories or goals nor tools usage, no API calls, nor responses nor any data that is being processed by the agents, nor any secrets and env vars.

Data collected includes:

- Version of squadAI
  - So we can understand how many users are using the latest version
- Version of Python
  - So we can decide on what versions to better support
- General OS (e.g. number of CPUs, macOS/Windows/Linux)
  - So we know what OS we should focus on and if we could build specific OS related features
- Number of agents and tasks in a squad
  - So we make sure we are testing internally with similar use cases and educate people on the best practices
- Squad Process being used
  - Understand where we should focus our efforts
- If Agents are using memory or allowing delegation
  - Understand if we improved the features or maybe even drop them
- If Tasks are being executed in parallel or sequentially
  - Understand if we should focus more on parallel execution
- Language model being used
  - Improved support on most used languages
- Roles of agents in a squad
  - Understand high level use cases so we can build better tools, integrations and examples about it
- Tools names available
  - Understand out of the publically available tools, which ones are being used the most so we can improve them

Users can opt-in sharing the complete telemetry data by setting the `share_squad` attribute to `True` on their Squads.

## License

SquadAI is released under the MIT License.
