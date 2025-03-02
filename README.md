# SalesGuru Crew

Welcome to the SalesGuru Crew project, powered by [crewAI](https://crewai.com). This template is designed to help you set up a multi-agent AI system with ease, leveraging the powerful and flexible framework provided by crewAI. Our goal is to enable your agents to collaborate effectively on complex tasks, maximizing their collective intelligence and capabilities.

## Installation

Ensure you have Python >=3.10 <3.13 installed on your system. This project uses [UV](https://docs.astral.sh/uv/) for dependency management and package handling, offering a seamless setup and execution experience.

First, if you haven't already, install uv:

```bash
pip install uv
```

Next, navigate to your project directory and install the dependencies:

(Optional) Lock the dependencies and install them by using the CLI command:
```bash
crewai install
```
### Customizing

**Add your `OPENAI_API_KEY` into the `.env` file**

- Modify `src/sales_guru/config/agents.yaml` to define your agents
- Modify `src/sales_guru/config/tasks.yaml` to define your tasks
- Modify `src/sales_guru/crew.py` to add your own logic, tools and specific args
- Modify `src/sales_guru/main.py` to add custom inputs for your agents and tasks

## Running the Project

To kickstart your crew of AI agents and begin task execution, run this from the root folder of your project:

```bash
$ crewai run
```

You can also use additional commands for different modes of operation:

```bash
$ crewai run train  # Run in training mode with multiple iterations
$ crewai run replay # Replay a specific task execution
$ crewai run test   # Test your crew with specific parameters
```

When running the SalesGuru system, you'll be prompted to input:
- Your company name
- Your company description

For training and testing modes, you'll also specify:
- Number of iterations
- Filename to save results (training mode)
- OpenAI model to use (testing mode)

## Current Functionality

### Active Agents

SalesGuru currently implements the following active AI agents:

1. **Lead Qualification and Fit Analysis Agent**
   - Analyzes leads to determine their fit with your company's offerings
   - Evaluates industry alignment, growth potential, use cases, decision-maker influence, and technology compatibility
   - Assigns scores and classifications (HOT, WARM, or COLD) to prioritize sales efforts

2. **Silent Task Completion Monitor (Supervisor)**
   - Ensures task completion and quality by monitoring other agents
   - Delegates incomplete tasks back to original agents with specific feedback
   - Operates behind the scenes for quality assurance

### Tools

The system comes equipped with several tools:

- **Web Search Tool** (SerperDevTool): For gathering online information about leads
- **CSV Search Tool**: For searching through lead data
- **File Read Tool**: For accessing lead information
- **Task Validator Tool**: For ensuring tasks are completed correctly

### Task Workflow

The current implementation focuses on lead qualification with a structured workflow:

1. The system takes in your company details as input
2. The Lead Qualification Agent analyzes each lead against your company profile
3. Each lead receives a score (0-100), classification, and recommendations
4. Results are formatted as JSON, ready for import into your sales process
5. The Supervisor Agent ensures all tasks are completed properly

## Future Expansion

SalesGuru is designed for future expansion with several additional agents already configured in the codebase but currently commented out:

- Prospect Research Agent
- Email & Outreach Agent
- Sales Call Prep Agent
- Objection Handling Agent
- Proposal & Quote Generator Agent
- Competitor Analysis Agent
- Sales Forecasting Agent
- Follow-up & Nurture Agent
- Customer Success & Upsell Agent

## Understanding Your Crew

The SalesGuru Crew is composed of multiple AI agents, each with unique roles, goals, and tools. These agents collaborate on a series of tasks, defined in `config/tasks.yaml`, leveraging their collective skills to achieve complex objectives. The `config/agents.yaml` file outlines the capabilities and configurations of each agent in your crew.

## Support

For support, questions, or feedback regarding the SalesGuru Crew or crewAI.
- Visit our [documentation](https://docs.crewai.com)
- Reach out to us through our [GitHub repository](https://github.com/joaomdmoura/crewai)
- [Join our Discord](https://discord.com/invite/X4JWnZnxPb)
- [Chat with our docs](https://chatg.pt/DWjSBZn)

Let's create wonders together with the power and simplicity of crewAI.
