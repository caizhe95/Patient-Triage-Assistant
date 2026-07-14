  
from __future__ import annotations  
 
from functools import lru_cache  
 
from langgraph.graph import END, StateGraph  
 
from app.graph.nodes import generate_answer_node, llm_planner_node, retrieve_node, safety_guard_node
from app.graph.state import TriageState  
 
 
def _safety_branch(state: TriageState) -> str:
    return 'blocked' if state.get('is_emergency') else 'continue'
 
 
def build_workflow():  
    graph = StateGraph(TriageState)  
    graph.add_node('safety_guard_node', safety_guard_node)
    graph.add_node('llm_planner_node', llm_planner_node)
    graph.add_node('retrieve_node', retrieve_node)  
    graph.add_node('generate_answer_node', generate_answer_node)  
    graph.set_entry_point('safety_guard_node')
    graph.add_conditional_edges('safety_guard_node', _safety_branch, {'blocked': END, 'continue': 'llm_planner_node'})
    graph.add_edge('llm_planner_node', 'retrieve_node')
    graph.add_edge('retrieve_node', 'generate_answer_node')  
    graph.add_edge('generate_answer_node', END)  
    return graph.compile()  
 
 
@lru_cache(maxsize=1)  
def get_workflow():  
    return build_workflow()  
