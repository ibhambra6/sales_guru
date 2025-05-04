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

## Technical Architecture

SalesGuru is built on the crewAI framework, employing a sophisticated multi-agent system with robust error handling, rate limiting, and task validation mechanisms. The architecture is designed to ensure reliable and consistent task completion, even in the face of API failures or model limitations.

### Core Components

#### 1. Agent Framework

The agent system is defined in YAML configuration (`src/sales_guru/config/agents.yaml`) for easy customization without modifying code. The system implements:

- **Role-Based Agent Design**: Each agent has a clearly defined role, goal, and backstory
- **Hierarchical Process Model**: Using CrewAI's hierarchical process with a supervisor agent
- **Task Dependency Chain**: Tasks are structured with explicit dependencies on previous task outputs

#### 2. Task Completion Architecture

A major technical feature is the robust task completion guarantee system:

- **TaskCompletionMonitor**: A sophisticated monitoring system that tracks task execution, detects failures, and handles retries
- **RateLimitedLLM**: A custom LLM wrapper that implements:
  - Exponential backoff with jitter for rate limit handling
  - Comprehensive error detection and recovery
  - Empty response detection and fallback generation
  - Token usage tracking to stay under API rate limits

#### 3. Error Resilience Subsystem

Multiple layers of error handling ensure system reliability:

- **Function-Level Error Handling**: Try/except blocks with specific error type handling
- **Task-Level Monitoring**: `@ensure_task_completion` decorator for task-level guarantees
- **System-Level Recovery**: Global retry logic in main.py with exponential backoff
- **Model-Level Patches**: Monkey patches to the CrewAI library to prevent empty response errors

#### 4. Data Flow Architecture

The system uses a structured data flow pattern:

- **Configuration-Driven**: YAML files for agent and task definition
- **Progressive Enhancement**: Each task takes previous task outputs as context
- **Validated Output Structure**: JSON output with consistent schema validation

### Code Organization

The codebase is structured for modularity and maintainability:

```
src/sales_guru/
├── config/                # Configuration files
│   ├── agents.yaml       # Agent definitions with roles, goals, backstories
│   └── tasks.yaml        # Task definitions with descriptions and expected outputs
├── tools/                 # Custom tools
│   ├── task_validator.py # Tool for validating task outputs
│   └── custom_tool.py    # Base class for custom tools
├── crew.py               # Main crew definition and agent initialization
├── task_monitor.py       # Task monitoring and completion guarantee system
├── supervisors.py        # Supervisor agent implementation
└── main.py               # CLI entry point and global error handling
```

## Agentic Framework Implementation

The SalesGuru system employs a sophisticated agentic framework with several key technical features:

### 1. Agent Enhancement System

Each agent in the system is "enhanced" through the `task_monitor.enhance_agent()` method, which adds:

- Token usage monitoring
- Rate limit awareness
- Error recovery capabilities
- Completion guarantees

```python
@agent
def lead_qualification(self) -> Agent:
    agent = Agent(
        config=self.agents_config['lead_qualification'],
        tools=[csv_search_tool, csv_read_tool, web_search_tool, task_validator_tool],
        verbose=True,
        llm=google_llm
    )
    # Enhance the agent with completion guarantees
    return self.task_monitor.enhance_agent(agent)
```

### 2. Multi-Model LLM Strategy

SalesGuru implements a strategic multi-model approach:

- **Google Gemini**: Used for lead qualification and research tasks (`gemini/gemini-2.0-flash`)
- **OpenAI GPT-4o**: Used for content generation and sales material preparation
- **Custom RateLimitedLLM Wrapper**: Provides consistent interfaces and error handling

```python
# Initialize the rate-limited LLMs
google_llm = RateLimitedLLM(
    model="gemini/gemini-2.0-flash",
    api_key=google_api_key,
    temperature=0.7
)

openai_llm = RateLimitedLLM(
    model="gpt-4o",
    api_key=openai_api_key,
    temperature=0.7
)
```

### 3. Task Validation Architecture

Tasks are validated through a comprehensive system:

- **TaskValidatorTool**: A specialized tool that verifies outputs match requirements
- **JSON Schema Validation**: Checking output structure against expected patterns
- **Completeness Scoring**: Tasks are scored from 0-1 on meeting requirements
- **Specific Issue Identification**: Pinpointing exactly what's missing from outputs

### 4. CrewAI Integration

The system uses the CrewAI framework with several enhancements:

- **Task Dependencies**: Tasks depend on previous tasks' outputs via the context parameter
- **Hierarchical Process**: Using CrewAI's hierarchical process with manager supervision
- **Custom Monkey Patches**: Patching CrewAI methods for better error handling

## Current Agents in Detail

### 1. Lead Qualification Agent

**Technical Implementation:**
- **Model**: Google Gemini 2.0 Flash
- **Tools**: CSV Search, CSV Read, Web Search, Task Validator
- **Input**: Company details and lead CSV data
- **Output**: JSON-structured lead scoring with classifications

This agent applies sophisticated analysis using a 0-100 scoring system and HOT/WARM/COLD classification based on:
- Industry alignment with company's target markets
- Company size and growth potential
- Potential use cases for products/services
- Decision-maker seniority and influence
- Current tech stack compatibility

### 2. Prospect Research Agent

**Technical Implementation:**
- **Model**: Google Gemini 2.0 Flash
- **Tools**: Web Search, Task Validator, Web Scraping
- **Input**: Qualified leads from previous agent
- **Output**: Enriched lead data with detailed company and contact information

This agent intelligently:
- Focuses only on HOT and WARM leads (ignoring COLD)
- Allocates research time proportionally based on lead priority
- Uses time-boxed web searches with max 2-3 searches per lead
- Applies "good enough" principle to avoid diminishing returns

### 3. Email Outreach Agent

**Technical Implementation:**
- **Model**: OpenAI GPT-4o
- **Tools**: Task Validator
- **Input**: Enriched lead data from Prospect Research Agent
- **Output**: Personalized email templates with subject lines, body content, and follow-up timing

This agent creates highly personalized outreach by:
- Crafting eye-catching subject lines referencing specific challenges
- Personalizing opening lines based on recent company news
- Articulating value propositions in context of specific business needs
- Including relevant social proof for the lead's industry
- Creating natural, conversational language with clear CTAs

### 4. Sales Call Preparation Agent

**Technical Implementation:**
- **Model**: OpenAI GPT-4o
- **Tools**: Task Validator
- **Input**: Combined data from all previous agents
- **Output**: One-page call briefs for sales representatives

This agent synthesizes critical information into:
- Concise company snapshots
- Decision-maker profiles
- Strategic talking points
- Anticipated objections with prepared responses
- Clear next steps and desired outcomes

### 5. Silent Task Completion Monitor (Supervisor)

**Technical Implementation:**
- **Model**: Google Gemini 2.0 Flash
- **Role**: Manager in hierarchical process
- **Function**: Ensures task completion without user-facing output

This agent:
- Delegates incomplete tasks back to original agents
- Provides specific feedback on missing requirements
- Operates completely behind the scenes
- Never executes tasks itself or provides direct feedback to users

## Task Workflow and Dependencies

The system implements a carefully structured task dependency chain:

1. **Lead Qualification Task**
   - Takes company details as input
   - Analyzes leads against company profile
   - Outputs scored and classified leads

2. **Prospect Research Task**
   - Takes qualified leads as input (context=[lead_qualification_task])
   - Researches HOT and WARM leads only
   - Outputs enriched lead data

3. **Email Outreach Task**
   - Takes enriched leads as input (context=[prospect_research_task, lead_qualification_task])
   - Creates personalized email templates
   - Outputs ready-to-send emails

4. **Sales Call Prep Task**
   - Takes all previous data as input (context=[email_outreach_task, prospect_research_task, lead_qualification_task])
   - Creates one-page call briefs
   - Outputs comprehensive sales preparation materials

Each task builds upon previous tasks, creating a progressive enhancement of lead data throughout the workflow.

## Tools

The system comes equipped with several specialized tools:

- **Web Search Tool (SerperDevTool)**: For gathering online information about leads and companies
- **CSV Search/Read Tools**: For accessing and querying lead data from CSV files
- **Task Validator Tool**: For ensuring outputs meet all requirements
- **Website Scraping Tool**: For extracting structured data from company websites

## Future Expansion

SalesGuru is designed for future expansion with several additional agents already configured in the codebase but currently commented out:

- Objection Handling Agent
- Proposal & Quote Generator Agent
- Competitor Analysis Agent
- Sales Forecasting Agent
- Follow-up & Nurture Agent
- Customer Success & Upsell Agent

The modular architecture makes adding these agents a straightforward process of uncommenting the relevant code and implementing their specific tools and logic.

## Support

For support, questions, or feedback regarding the SalesGuru Crew or crewAI.
- Visit our [documentation](https://docs.crewai.com)
- Reach out to us through our [GitHub repository](https://github.com/joaomdmoura/crewai)
- [Join our Discord](https://discord.com/invite/X4JWnZnxPb)
- [Chat with our docs](https://chatg.pt/DWjSBZn)

Let's create wonders together with the power and simplicity of crewAI.
