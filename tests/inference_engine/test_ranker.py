from src.inference_engine.ranker import AnswerRanker

def test_ranker_heuristic():
    ranker = AnswerRanker()
    candidates = ["short", "longer answer", "mid"]
    context = {}
    ranked = ranker.rank_answers(candidates, context)
    assert ranked[0][0] == "longer answer"
    assert len(ranked) == 3

def test_ranker_empty():
    ranker = AnswerRanker()
    candidates = []
    context = {}
    ranked = ranker.rank_answers(candidates, context)
    assert ranked == []
