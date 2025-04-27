from src.data_pipeline.text_perturb import synonym_replacement

def test_synonym_replacement():
    text = "The cat sat on the mat."
    perturbed = synonym_replacement(text)
    assert isinstance(perturbed, str)
    assert len(perturbed) > 0
