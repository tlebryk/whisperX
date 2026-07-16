import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from whisperx import alignment


class FixedEmissionModel(torch.nn.Module):
    """Small deterministic torchaudio-style model for alignment tests."""

    def forward(self, waveform, lengths=None):
        logits = torch.tensor(
            [
                [
                    [5.0, 0.0, 0.0],
                    [0.0, 5.0, 0.0],
                    [0.0, 5.0, 0.0],
                    [5.0, 0.0, 0.0],
                    [0.0, 0.0, 5.0],
                    [5.0, 0.0, 0.0],
                ]
            ],
            device=waveform.device,
        ).repeat(waveform.shape[0], 1, 1)
        output_lengths = torch.full(
            (waveform.shape[0],),
            logits.shape[1],
            dtype=torch.long,
            device=waveform.device,
        )
        return logits, output_lengths


ALIGN_METADATA = {
    "dictionary": {"<pad>": 0, "a": 1, "b": 2},
    "language": "en",
    "type": "torchaudio",
}


def test_serial_alignment_matches_whisperx_3_4_4_golden_output():
    result = alignment.align(
        transcript=[{"start": 0.0, "end": 0.06, "text": "ab"}],
        model=FixedEmissionModel(),
        align_model_metadata=ALIGN_METADATA,
        audio=torch.zeros(960),
        device="cpu",
        return_char_alignments=True,
    )

    assert result == {
        "segments": [
            {
                "start": 0.01,
                "end": 0.03,
                "text": "ab",
                "words": [
                    {
                        "word": "ab",
                        "start": 0.01,
                        "end": 0.03,
                        "score": 0.497,
                    }
                ],
                "chars": [
                    {"char": "a", "start": 0.01, "end": 0.02, "score": 0.987},
                    {"char": "b", "start": 0.02, "end": 0.03, "score": 0.007},
                ],
            }
        ],
        "word_segments": [
            {"word": "ab", "start": 0.01, "end": 0.03, "score": 0.497}
        ],
    }


@pytest.mark.parametrize(
    ("method", "expected"),
    [
        ("nearest", [1.0, 1.0, 1.0, 3.0]),
        ("linear", [1.0, 1.0, 2.0, 3.0]),
        ("ignore", [np.nan, 1.0, np.nan, 3.0]),
    ],
)
def test_interpolate_missing(method, expected):
    actual = alignment._interpolate_missing([np.nan, 1.0, np.nan, 3.0], method)
    np.testing.assert_equal(actual, expected)


def test_align_batch_sorts_by_duration_and_restores_input_order(monkeypatch):
    acoustic_batches = []

    def compute_batch(waveforms, model, model_type, device):
        acoustic_batches.append([waveform.shape[-1] for waveform in waveforms])
        return [torch.empty(1, 3) for _ in waveforms]

    def finish(segment, *args, **kwargs):
        return [
            {
                "start": segment["start"],
                "end": segment["end"],
                "text": segment["text"],
                "words": [],
            }
        ]

    monkeypatch.setattr(alignment, "_compute_emission_batch", compute_batch)
    monkeypatch.setattr(alignment, "_finish_alignment_segment", finish)

    transcripts = [
        [{"start": 0.0, "end": 0.03, "text": "a"}],
        [{"start": 0.0, "end": 0.01, "text": "a"}],
        [{"start": 0.0, "end": 0.02, "text": "b"}],
    ]
    results = alignment.align_batch(
        transcripts=transcripts,
        model=None,
        align_model_metadata=ALIGN_METADATA,
        audio=[torch.zeros(480), torch.zeros(160), torch.zeros(320)],
        device="cpu",
        batch_size=2,
    )

    assert acoustic_batches == [[160, 320], [480]]
    assert [result["segments"][0]["text"] for result in results] == ["a", "a", "b"]


def test_align_batch_size_one_matches_serial_when_emissions_match(monkeypatch):
    logits, _ = FixedEmissionModel()(torch.zeros(1, 960))
    emission = torch.log_softmax(logits[0], dim=-1)

    monkeypatch.setattr(
        alignment,
        "_compute_emission",
        lambda waveform, model, model_type, device: emission,
    )
    monkeypatch.setattr(
        alignment,
        "_compute_emission_batch",
        lambda waveforms, model, model_type, device: [emission for _ in waveforms],
    )

    transcript = [{"start": 0.0, "end": 0.06, "text": "ab"}]
    serial = alignment.align(
        transcript=transcript,
        model=None,
        align_model_metadata=ALIGN_METADATA,
        audio=torch.zeros(960),
        device="cpu",
        return_char_alignments=True,
    )
    batched = alignment.align_batch(
        transcripts=[transcript],
        model=None,
        align_model_metadata=ALIGN_METADATA,
        audio=[torch.zeros(960)],
        device="cpu",
        batch_size=1,
        return_char_alignments=True,
    )[0]

    assert batched == serial


def test_align_batch_validates_inputs():
    with pytest.raises(ValueError, match="same number"):
        alignment.align_batch(
            transcripts=[],
            model=None,
            align_model_metadata=ALIGN_METADATA,
            audio=[torch.zeros(1)],
            device="cpu",
            batch_size=1,
        )

    with pytest.raises(ValueError, match="at least 1"):
        alignment.align_batch(
            transcripts=[],
            model=None,
            align_model_metadata=ALIGN_METADATA,
            audio=[],
            device="cpu",
            batch_size=0,
        )


@pytest.mark.parametrize("feat_extract_norm", ["group", None, "custom"])
def test_huggingface_batch_emissions_fall_back_for_unvalidated_norms(
    monkeypatch, feat_extract_norm
):
    class UnsupportedBatchModel:
        config = SimpleNamespace(feat_extract_norm=feat_extract_norm)

        def __call__(self, waveform):
            raise AssertionError("the padded Hugging Face batch path must not run")

    serial_calls = []

    def compute_serial(waveform, model, model_type, device):
        serial_calls.append((waveform.clone(), model, model_type, device))
        return torch.full((waveform.shape[-1], 2), len(serial_calls), dtype=torch.float32)

    monkeypatch.setattr(alignment, "_compute_emission", compute_serial)
    model = UnsupportedBatchModel()
    waveforms = [torch.zeros(1, 399), torch.zeros(800)]

    emissions = alignment._compute_emission_batch(
        waveforms,
        model,
        "huggingface",
        "cpu",
    )

    assert [emission.shape for emission in emissions] == [(399, 2), (800, 2)]
    assert [call[0].shape for call in serial_calls] == [(1, 399), (1, 800)]
    assert all(call[1:] == (model, "huggingface", "cpu") for call in serial_calls)
    assert emissions[0][0, 0].item() == 1
    assert emissions[1][0, 0].item() == 2


@pytest.mark.skipif(
    os.environ.get("WHISPERX_RUN_MODEL_TESTS") != "1",
    reason="downloads and runs the 360 MB torchaudio wav2vec2 model",
)
def test_real_torchaudio_batch_emissions_match_serial():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, metadata = alignment.load_align_model("en", device)
    model.eval()
    waveforms = [
        torch.linspace(-0.1, 0.1, 399).unsqueeze(0),
        torch.sin(torch.linspace(0, 100, 8000)).unsqueeze(0),
        torch.sin(torch.linspace(0, 200, 16000)).unsqueeze(0),
    ]

    serial = [
        alignment._compute_emission(waveform, model, metadata["type"], device)
        for waveform in waveforms
    ]
    batched = alignment._compute_emission_batch(
        waveforms, model, metadata["type"], device
    )

    assert [item.shape for item in batched] == [item.shape for item in serial]
    for actual, expected in zip(batched, serial):
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
