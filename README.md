# LLM Agent Pipeline with LangGraph & Groq LLaMA3

A lightweight, modular LLM agent framework built using LangGraph, powered by Groq's ultra-fast LLaMA3-8B model. Demonstrates goal-oriented agents that break down user prompts into sub-tasks, execute them using LLM reasoning, and provide structured summaries through graph-based flow.

## Features

- Modular graph architecture using LangGraph
- Ultra-fast LLM responses with Groq's llama3-8b-8192
- Dynamic planning, execution, and summarization workflow
- Clean output at each step for inspection and reuse
- Extensible framework for tools, memory, and custom nodes

## Architecture Flow

```
    User Input
        ↓
   Planner Node
        ↓
  Executor Node
        ↓
 Summarizer Node
        ↓
   Final Output
```

### LangGraph Flow Pattern

The project implements a **StateGraph** pattern with three main nodes:

- **Planner Node**: Analyzes user input and breaks down complex tasks into actionable sub-tasks
- **Executor Node**: Processes each sub-task using LLM reasoning and generates intermediate results
- **Summarizer Node**: Consolidates all outputs into a coherent final response

Each node maintains state context and can pass information to subsequent nodes, enabling complex multi-step reasoning workflows.

## Tech Stack

| Component | Purpose |
|-----------|---------|
| **LangGraph** | State-based flow control for LLM agents |
| **LangChain** | LLM interface abstraction |
| **Groq API** | High-speed inference with LLaMA3 |
| **Python** | Core logic and execution |
| **dotenv** | Secure API key management |

## Quick Start

### Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Configure environment
echo "GROQ_API_KEY=your_groq_api_key_here" > .env

# Run application
python main.py
```

## Example Usage

**Input:** "Plan and write a blog post on benefits of AI in education"

**Planner Output:**
- Research recent uses of AI in education
- Draft key benefits with examples
- Conclude with future outlook

**Final Output:** Comprehensive blog-ready content connecting all planned points

## Why LangGraph?

LangGraph enables building stateful, multi-step agents that mirror human reasoning processes. This project provides a practical implementation of LLM-powered pipelines with clear separation of planning, execution, and summarization phases.

## Extension Ideas

- Integrate web search capabilities (SerpAPI)
- Add persistent memory for long-term task tracking
- Convert to CLI application or Streamlit dashboard
- Implement custom tool integrations

## Requirements

- Python 3.8+
- Groq API key
- LangGraph and LangChain dependencies

## License

MIT License
