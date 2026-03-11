import operator
import asyncio
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# Shared in-memory checkpointer — enables interrupt/resume and full state persistence per thread
_checkpointer = MemorySaver()


class ResearchState(TypedDict):
    topic: str
    report_config: dict
    research_data: dict
    research_chunks: dict
    analysis: dict
    bias_report: str
    validation_report: str
    gaps: str
    qc_score: int
    qc_passed: bool
    qc_feedback: str
    qc_iterations: int
    planner_strategy: dict
    logs: Annotated[list[str], operator.add]


def researcher_node(state: ResearchState) -> dict:
    from agents import ResearcherAgent

    agent = ResearcherAgent()
    topic = state["topic"]
    report_config = state.get("report_config", {})
    qc_iterations = state.get("qc_iterations", 0)
    qc_feedback = state.get("qc_feedback", "")

    context = {
        "topic": topic,
        "uploaded_files": report_config.get("uploaded_files", []),
        "qc_retry": qc_iterations > 0 or bool(qc_feedback),
        "qc_feedback": qc_feedback,
        "planner_strategy": state.get("planner_strategy", {}),
        **report_config
    }

    result = agent.perform_task(context)
    research_chunks = result.pop("_chunks", {})

    return {
        "research_data": result,
        "research_chunks": research_chunks,
        "logs": agent.logs
    }


def analyst_node(state: ResearchState) -> dict:
    from agents import AnalystAgent

    agent = AnalystAgent()
    result = agent.perform_task(state["research_data"])
    return {"analysis": result, "logs": agent.logs}


async def parallel_critics_node(state: ResearchState) -> dict:
    from agents import GapAnalystAgent, BiasDetectorAgent, FactCheckerAgent

    gap_agent = GapAnalystAgent()
    bias_agent = BiasDetectorAgent()
    fact_agent = FactCheckerAgent()

    context = {
        "topic": state["topic"],
        "research_data": state["research_data"],
        **state.get("report_config", {})
    }

    gap_result, bias_result, fact_result = await asyncio.gather(
        gap_agent.perform_task_async(context),
        bias_agent.perform_task_async(context),
        fact_agent.perform_task_async(context)
    )

    merged = {}
    merged.update(gap_result)
    merged.update(bias_result)
    merged.update(fact_result)
    merged["logs"] = gap_agent.logs + bias_agent.logs + fact_agent.logs
    return merged


def synthesizer_node(state: ResearchState) -> dict:
    from agents import SynthesizerAgent

    agent = SynthesizerAgent()
    agent.perform_task({})
    return {"logs": agent.logs}


def qc_node(state: ResearchState) -> dict:
    from agents import QualityControlAgent

    agent = QualityControlAgent()
    context = {
        "topic": state["topic"],
        "research_data": state["research_data"],
        "analysis": state["analysis"],
        "qc_iterations": state.get("qc_iterations", 0)
    }
    result = agent.perform_task(context)
    new_iterations = state.get("qc_iterations", 0) + 1

    return {
        "qc_score": result.get("score", 0),
        "qc_passed": result.get("qc_passed", True),
        "qc_feedback": result.get("feedback", ""),
        "qc_iterations": new_iterations,
        "logs": agent.logs
    }


def formatter_node(state: ResearchState) -> dict:
    from agents import FormatterAgent

    agent = FormatterAgent()
    agent.perform_task({})
    return {"logs": agent.logs}


def planner_node(state: ResearchState) -> dict:
    from agents import PlannerAgent
    agent = PlannerAgent()
    context = {
        "topic": state["topic"],
        "qc_feedback": state.get("qc_feedback", ""),
        "gaps": state.get("gaps", ""),
    }
    result = agent.perform_task(context)
    return {"planner_strategy": result.get("planner_strategy", {}), "logs": agent.logs}


def should_loop_back(state: ResearchState) -> str:
    qc_passed = state.get("qc_passed", True)
    qc_iterations = state.get("qc_iterations", 0)

    if not qc_passed and qc_iterations < 2:
        return "planner"
    else:
        return "formatter"


def build_research_graph(interrupt: bool = True):
    graph = StateGraph(ResearchState)

    graph.add_node("researcher", researcher_node)
    graph.add_node("analyst", analyst_node)
    graph.add_node("parallel_critics", parallel_critics_node)
    graph.add_node("synthesizer", synthesizer_node)
    graph.add_node("qc", qc_node)
    graph.add_node("planner", planner_node)
    graph.add_node("formatter", formatter_node)

    graph.add_edge("researcher", "analyst")
    graph.add_edge("analyst", "parallel_critics")
    graph.add_edge("parallel_critics", "synthesizer")
    graph.add_edge("synthesizer", "qc")
    graph.add_conditional_edges(
        "qc",
        should_loop_back,
        {"planner": "planner", "formatter": "formatter"}
    )
    graph.add_edge("planner", "researcher")
    graph.add_edge("formatter", END)

    graph.set_entry_point("researcher")

    # Conditionally pause before formatting for Human-in-the-Loop
    if interrupt:
        return graph.compile(
            checkpointer=_checkpointer,
            interrupt_before=["formatter"]
        )
    return graph.compile(checkpointer=_checkpointer)


research_graph = build_research_graph()
