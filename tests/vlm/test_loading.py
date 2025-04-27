import pytest
from src.vlm import loading

def test_load_vlm_model_monkeypatch(monkeypatch):
    # Patch out actual model loading for test speed
    class DummyProcessor:
        pass
    class DummyModel:
        def to(self, device):
            return self
        def eval(self):
            return self
    monkeypatch.setattr(loading, "AutoProcessor", type("AutoProcessor", (), {"from_pretrained": staticmethod(lambda *a, **kw: DummyProcessor())}))
    monkeypatch.setattr(loading, "AutoModelForVision2Seq", type("AutoModelForVision2Seq", (), {"from_pretrained": staticmethod(lambda *a, **kw: DummyModel())}))
    monkeypatch.setattr(loading, "load_config", lambda: {"vlm": {"model_id": "mock", "device": "cpu"}})
    model, processor = loading.load_vlm_model()
    assert model is not None
    assert processor is not None
