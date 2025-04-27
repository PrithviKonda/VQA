from src.knowledge.planner import Planner

def test_plan_basic():
    planner = Planner()
    steps = planner.plan("What is the color and shape?", {})
    assert steps[0] == "decompose_question"
    assert steps[-1] == "generate_final_answer"

def test_plan_empty():
    planner = Planner()
    steps = planner.plan("", {})
    assert steps == ["invalid_question"]
