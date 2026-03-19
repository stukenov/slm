import torch
from kzcalm.model.duration_predictor import DurationPredictor


def test_duration_predictor_shape():
    B, S, D = 2, 10, 256
    pred = DurationPredictor(d_model=D)
    x = torch.randn(B, S, D)
    mask = torch.ones(B, S)
    log_dur = pred(x, mask)
    assert log_dur.shape == (B, S)


def test_duration_predictor_masked():
    B, S, D = 1, 5, 256
    pred = DurationPredictor(d_model=D)
    x = torch.randn(B, S, D)
    mask = torch.tensor([[1, 1, 1, 0, 0]], dtype=torch.float)
    log_dur = pred(x, mask)
    assert (log_dur[0, 3:] == 0).all()
