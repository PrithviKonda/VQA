from src.inference_engine.ranker import Ranker

def test_ranker_heuristic():
    ranker = Ranker()
    candidates = ["short", "longer answer", "mid"]
    context = {}
    ranked = ranker.rank(candidates, context)
    assert ranked[0][0] == "longer answer"
    assert len(ranked) == 3

def test_ranker_empty():
    ranker = Ranker()
    candidates = []
    context = {}
    ranked = ranker.rank(candidates, context)
    assert ranked == []
