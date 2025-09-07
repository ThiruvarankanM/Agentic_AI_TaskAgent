from typing import TypedDict, Optional, List, Dict, Any
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
import re
import json
import requests
import sqlite3
from datetime import datetime
import subprocess
import time

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    raise ValueError("Need GROQ_API_KEY in .env file")

llm = ChatGroq(temperature=0.2, groq_api_key=groq_api_key, model_name="llama3-8b-8192")

class AgentState(TypedDict):
    goal: str
    plan: Optional[List[Dict]]
    current_step: Optional[int]
    results: Optional[List[Dict]]
    memory: Optional[Dict]
    tools_used: Optional[List[str]]
    confidence: Optional[float]
    need_human: Optional[bool]

class Memory:
    def __init__(self):
        self.db = sqlite3.connect('agent_memory.db', check_same_thread=False)
        self.setup_db()
    
    def setup_db(self):
        self.db.execute('''
            CREATE TABLE IF NOT EXISTS experiences (
                id INTEGER PRIMARY KEY,
                goal TEXT,
                success INTEGER,
                approach TEXT,
                timestamp TEXT
            )
        ''')
        self.db.commit()
    
    def save_experience(self, goal, success, approach):
        self.db.execute(
            'INSERT INTO experiences (goal, success, approach, timestamp) VALUES (?, ?, ?, ?)',
            (goal, 1 if success else 0, approach, datetime.now().isoformat())
        )
        self.db.commit()
    
    def get_similar_experiences(self, goal):
        cursor = self.db.execute(
            'SELECT * FROM experiences WHERE goal LIKE ? ORDER BY success DESC LIMIT 3',
            (f'%{goal}%',)
        )
        return cursor.fetchall()

class Tools:
    @staticmethod
    def web_search(query):
        try:
            # Simple search using DuckDuckGo (no API key needed)
            url = f"https://api.duckduckgo.com/?q={query}&format=json&no_html=1&skip_disambig=1"
            response = requests.get(url, timeout=10)
            data = response.json()
            
            if data.get('AbstractText'):
                return data['AbstractText']
            elif data.get('RelatedTopics') and len(data['RelatedTopics']) > 0:
                return data['RelatedTopics'][0].get('Text', 'No results found')
            else:
                return 'No useful results found'
        except:
            return 'Search failed - network issue'
    
    @staticmethod
    def run_code(code, language='python'):
        try:
            if language == 'python':
                with open('temp_code.py', 'w') as f:
                    f.write(code)
                result = subprocess.run(['python', 'temp_code.py'], 
                                      capture_output=True, text=True, timeout=30)
                os.remove('temp_code.py')
                return result.stdout if result.returncode == 0 else result.stderr
            else:
                return 'Only Python supported for now'
        except:
            return 'Code execution failed'
    
    @staticmethod
    def write_file(filename, content):
        try:
            with open(filename, 'w') as f:
                f.write(content)
            return f'File {filename} created successfully'
        except Exception as e:
            return f'Failed to write file: {str(e)}'
    
    @staticmethod
    def read_file(filename):
        try:
            with open(filename, 'r') as f:
                return f.read()
        except Exception as e:
            return f'Failed to read file: {str(e)}'

memory = Memory()
tools = Tools()

def smart_planner(state: AgentState) -> AgentState:
    goal = state['goal']
    
    # Check past experiences
    similar = memory.get_similar_experiences(goal)
    experience_context = ""
    if similar:
        experience_context = f"\nPast similar tasks: {len(similar)} found, {len([x for x in similar if x[2] == 1])} successful"
    
    prompt = f"""
    Goal: {goal}
    {experience_context}
    
    Create a smart plan with these rules:
    1. Each step should specify WHAT to do and WHICH tool to use
    2. Available tools: web_search, run_code, write_file, read_file, think
    3. Plan can be 2-7 steps depending on complexity
    4. Each step should build on previous ones
    5. Include a confidence check step at the end
    
    Format as JSON:
    [{{"step": 1, "action": "what to do", "tool": "which tool", "why": "reasoning"}}, ...]
    """
    
    try:
        response = llm.invoke(prompt).content
        # Extract JSON from response
        json_match = re.search(r'\[.*\]', response, re.DOTALL)
        if json_match:
            plan = json.loads(json_match.group())
        else:
            # Fallback plan
            plan = [
                {"step": 1, "action": f"Research information about {goal}", "tool": "web_search", "why": "Need background info"},
                {"step": 2, "action": f"Analyze findings and create approach", "tool": "think", "why": "Process information"},
                {"step": 3, "action": f"Execute solution for {goal}", "tool": "think", "why": "Complete the task"}
            ]
        
        print(f"[PLANNER] Created {len(plan)} step plan")
        return {"plan": plan, "current_step": 0, "confidence": 0.7}
    
    except Exception as e:
        print(f"[PLANNER] Error: {e}")
        fallback = [{"step": 1, "action": f"Complete {goal}", "tool": "think", "why": "Simple fallback"}]
        return {"plan": fallback, "current_step": 0, "confidence": 0.3}

def adaptive_executor(state: AgentState) -> AgentState:
    plan = state.get('plan', [])
    results = []
    tools_used = []
    
    for step_info in plan:
        step_num = step_info['step']
        action = step_info['action']
        tool = step_info['tool']
        
        print(f"\n[EXECUTOR] Step {step_num}: {action}")
        print(f"Using tool: {tool}")
        
        # Execute based on tool
        if tool == 'web_search':
            # Extract search query from action
            query = action.split('about')[-1].strip() if 'about' in action else action
            result = tools.web_search(query)
            tools_used.append('web_search')
            
        elif tool == 'run_code':
            # Ask LLM to generate code for the action
            code_prompt = f"Write Python code to: {action}\nJust return the code, nothing else."
            code = llm.invoke(code_prompt).content
            # Clean up code (remove markdown formatting)
            code = re.sub(r'```python\n?', '', code)
            code = re.sub(r'```\n?', '', code)
            result = tools.run_code(code)
            tools_used.append('run_code')
            
        elif tool == 'write_file':
            # Generate content and filename
            content_prompt = f"Create content for: {action}\nReturn just the content."
            content = llm.invoke(content_prompt).content
            filename = f"output_{step_num}.txt"
            result = tools.write_file(filename, content)
            tools_used.append('write_file')
            
        elif tool == 'read_file':
            # Extract filename from action or use default
            filename = 'output_1.txt'  # Simple default
            result = tools.read_file(filename)
            tools_used.append('read_file')
            
        else:  # 'think' or any other tool
            # Pure LLM reasoning
            think_prompt = f"""
            Task: {action}
            Context from previous steps: {results[-2:] if results else "None"}
            
            Provide a thoughtful response or solution.
            """
            result = llm.invoke(think_prompt).content
            tools_used.append('think')
        
        results.append({
            'step': step_num,
            'action': action,
            'tool': tool,
            'result': result,
            'timestamp': datetime.now().isoformat()
        })
        
        print(f"Result: {result[:100]}{'...' if len(result) > 100 else ''}")
        
        # Simple adaptive logic - if step failed, try alternative
        if 'error' in result.lower() or 'failed' in result.lower():
            print(f"Step {step_num} had issues, attempting alternative...")
            alt_result = llm.invoke(f"Alternative approach for: {action}").content
            results[-1]['alternative'] = alt_result
    
    return {"results": results, "tools_used": tools_used, "current_step": len(plan)}

def intelligent_summarizer(state: AgentState) -> AgentState:
    goal = state['goal']
    plan = state.get('plan', [])
    results = state.get('results', [])
    tools_used = state.get('tools_used', [])
    
    # Evaluate success
    success_indicators = ['completed', 'success', 'done', 'found', 'created']
    error_indicators = ['error', 'failed', 'unable', 'could not']
    
    total_results = len(results)
    successful_steps = 0
    
    for result in results:
        result_text = result['result'].lower()
        if any(indicator in result_text for indicator in success_indicators):
            successful_steps += 1
        elif any(indicator in result_text for indicator in error_indicators):
            successful_steps -= 0.5
    
    success_rate = max(0, successful_steps / total_results) if total_results > 0 else 0
    overall_success = success_rate > 0.6
    
    # Save experience
    approach_summary = f"Used tools: {', '.join(set(tools_used))}. {total_results} steps."
    memory.save_experience(goal, overall_success, approach_summary)
    
    prompt = f"""
    Goal: {goal}
    Success Rate: {success_rate:.2f}
    Tools Used: {', '.join(set(tools_used))}
    
    Results summary:
    """
    
    for i, result in enumerate(results, 1):
        prompt += f"\nStep {i}: {result['action']}\nOutcome: {result['result'][:200]}\n"
    
    prompt += f"""
    
    Provide a final summary including:
    1. Was the goal achieved? ({success_rate:.0%} success rate)
    2. What worked well?
    3. What could be improved?
    4. Key insights or deliverables
    5. Recommended next steps
    
    Be honest about limitations and partial successes.
    """
    
    try:
        summary = llm.invoke(prompt).content
        confidence = min(0.95, success_rate + 0.2)
        
        print(f"\n[SUMMARIZER] Task completed with {success_rate:.0%} success rate")
        
        return {
            "summary": summary,
            "confidence": confidence,
            "success_rate": success_rate,
            "need_human": success_rate < 0.4
        }
    
    except Exception as e:
        return {
            "summary": f"Completed {total_results} steps towards: {goal}. See individual results above.",
            "confidence": 0.5,
            "success_rate": success_rate,
            "need_human": True
        }

def create_advanced_agent():
    graph = StateGraph(AgentState)
    
    graph.add_node("planner", smart_planner)
    graph.add_node("executor", adaptive_executor)
    graph.add_node("summarizer", intelligent_summarizer)
    
    graph.set_entry_point("planner")
    graph.add_edge("planner", "executor")
    graph.add_edge("executor", "summarizer")
    graph.add_edge("summarizer", END)
    
    return graph.compile()

def main():
    print("Advanced Agentic AI System")
    print("Tools: web search, code execution, file operations")
    print("Features: memory, adaptation, confidence tracking")
    print("=" * 60)
    
    agent = create_advanced_agent()
    
    while True:
        try:
            goal = input("\nWhat should I help you accomplish? (or 'quit'): ").strip()
            
            if goal.lower() in ['quit', 'exit', 'q']:
                print("System shutdown.")
                break
            
            if not goal:
                continue
            
            print(f"\nProcessing: {goal}")
            print("-" * 50)
            
            start_time = time.time()
            result = agent.invoke({"goal": goal})
            duration = time.time() - start_time
            
            print("\n" + "=" * 60)
            print("FINAL REPORT")
            print("=" * 60)
            print(result["summary"])
            
            if result.get("need_human"):
                print(f"\n[WARNING] Low success rate ({result.get('success_rate', 0):.0%}). Human review recommended.")
            
            print(f"\nConfidence: {result.get('confidence', 0):.0%}")
            print(f"Duration: {duration:.1f} seconds")
            print(f"Tools used: {', '.join(result.get('tools_used', []))}")
            print("=" * 60)
            
        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye!")
            break
        except Exception as e:
            print(f"System error: {e}")

if __name__ == "__main__":
    main()
