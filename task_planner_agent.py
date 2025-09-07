from typing import TypedDict, Annotated, Optional, List
from langgraph.graph import StateGraph, END
from langchain_core.runnables import Runnable
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

# Load API key
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Use LLaMA3 from Groq
llm = ChatGroq(temperature=0, groq_api_key=groq_api_key, model_name="llama3-8b-8192")

# Define shared state
class AgentState(TypedDict):
    goal: str
    steps: Optional[List[str]]
    results: Optional[List[str]]
    summary: Optional[str]

# --- Nodes ---

# Planner node: split goal into 3 steps
def planner_node(state: AgentState) -> AgentState:
    prompt = f"Break down the goal '{state['goal']}' into exactly 3 simple steps."
    response = llm.invoke(prompt).content
    steps = [step.strip("- ").strip() for step in response.split("\n") if step.strip()]
    print("\n🧠 Planner Output:\n", steps)
    return {"steps": steps}

# Executor node: run each step using LLM
def executor_node(state: AgentState) -> AgentState:
    steps = state.get("steps", [])
    results = []
    for i, step in enumerate(steps):
        result = llm.invoke(f"Step {i+1}: {step}").content.strip()
        results.append(result)
    print("\n🤖 Executor Output:\n", results)
    return {"results": results}

# Summarizer node: summarize all results
def summarizer_node(state: AgentState) -> AgentState:
    steps = state.get("steps", [])
    results = state.get("results", [])
    prompt = "Summarize the following actions and their outcomes:\n\n"
    for i, (step, result) in enumerate(zip(steps, results), 1):
        prompt += f"{i}. {step}\n→ {result}\n\n"
    prompt += "Provide a short summary of the overall process."
    summary = llm.invoke(prompt).content.strip()
    print("\n📘 Summary Output:\n", summary)
    return {"summary": summary}

# --- LangGraph construction ---
graph = StateGraph(AgentState)
graph.add_node("planner", planner_node)
graph.add_node("executor", executor_node)
graph.add_node("summarizer", summarizer_node)

graph.set_entry_point("planner")
graph.add_edge("planner", "executor")
graph.add_edge("executor", "summarizer")
graph.add_edge("summarizer", END)  # Fixed: use END instead of set_finish_point

app = graph.compile()

# --- Run interaction ---
if __name__ == "__main__":
    print("LangGraph Agent System")
    user_goal = input("Enter your goal: ")
    initial_state = {"goal": user_goal}
    result = app.invoke(initial_state)
    print("\nFinal Summary:")
    print(result["summary"])
