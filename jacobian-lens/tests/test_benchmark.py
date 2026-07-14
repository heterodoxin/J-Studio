import pytest
import torch

from jlens.benchmark import (
    BenchmarkCase,
    ReadoutCase,
    find_token_position_containing,
    next_token_logits,
    rank_token,
    ranked_default_cases,
    render_html_report,
    render_readout_html_report,
    run_case,
    run_readout_case,
)
from jlens.lens import JacobianLens
from tests.tiny import TinyDecoder

PROMPT = "the quick brown fox jumps over the lazy dog " * 2


def test_rank_token_reports_zero_based_rank_and_logit():
    logits = torch.tensor([0.1, 3.0, -1.0, 2.0])

    rank = rank_token(logits, 3)

    assert rank.token_id == 3
    assert rank.rank == 1
    assert rank.logit == 2.0


def test_injection_case_records_strict_failed_noop():
    model = TinyDecoder(n_layers=4, d_model=8, seed=0)
    torch.manual_seed(10)
    lens = JacobianLens({2: torch.randn(8, 8) * 2}, n_prompts=1, d_model=8)
    case = BenchmarkCase(
        name="hard inject",
        operation="inject",
        prompt=PROMPT,
        target=20,
        layers=(2,),
        maximum_scale=1.0,
    )

    result = run_case(model, lens, case)

    assert not result.success
    assert result.applied is False
    assert result.target_after.rank == result.target_before.rank
    assert result.trace["message"] == "bounded search found no passing intervention"


def test_replacement_case_records_source_and_target_ranks():
    model = TinyDecoder(n_layers=4, d_model=8, seed=1)
    torch.manual_seed(2)
    lens = JacobianLens({2: torch.randn(8, 8)}, n_prompts=1, d_model=8)
    baseline = run_case(
        model,
        lens,
        BenchmarkCase(
            name="replace",
            operation="replace",
            prompt=PROMPT,
            source=1,
            target=2,
            layers=(2,),
            maximum_scale=0.25,
        ),
    )

    assert baseline.source_before is not None
    assert baseline.source_after is not None
    assert baseline.target_before.token_id == 2
    assert baseline.source_before.token_id == 1


def test_replacement_requires_nontrivial_overtake():
    model = TinyDecoder(n_layers=4, d_model=8, seed=1)
    torch.manual_seed(2)
    lens = JacobianLens({2: torch.randn(8, 8)}, n_prompts=1, d_model=8)
    logits = next_token_logits(model, PROMPT)
    top_ids = logits.argsort(descending=True)
    target = int(top_ids[0])
    source = int(top_ids[-1])

    result = run_case(
        model,
        lens,
        BenchmarkCase(
            name="trivial replace",
            operation="replace",
            prompt=PROMPT,
            source=source,
            target=target,
            layers=(2,),
            maximum_scale=16.0,
        ),
    )

    assert result.applied
    assert not result.success
    assert result.trace["benchmark_success"] is False
    assert "source must outrank target before replacement" in result.trace["benchmark_reason"]


def test_ranked_default_cases_are_nontrivial_baseline_choices():
    model = TinyDecoder(n_layers=4, d_model=8, seed=1)

    cases = ranked_default_cases(model, prompt=PROMPT, maximum_scale=8.0)

    assert [case.operation for case in cases] == ["inject", "replace", "replace"]
    assert cases[0].target == cases[1].target
    assert cases[1].source != cases[1].target
    assert cases[2].source != cases[2].target
    assert all(case.maximum_scale == 8.0 for case in cases)


def test_html_report_contains_summary_and_case_rows():
    model = TinyDecoder(n_layers=4, d_model=8, seed=0)
    torch.manual_seed(10)
    lens = JacobianLens({2: torch.randn(8, 8) * 2}, n_prompts=1, d_model=8)
    result = run_case(
        model,
        lens,
        BenchmarkCase(
            name="html inject",
            operation="inject",
            prompt=PROMPT,
            target=4,
            layers=(2,),
            maximum_scale=32.0,
        ),
    )

    page = render_html_report((result,), title="Smoke Report")

    assert "<!doctype html>" in page.lower()
    assert "Smoke Report" in page
    assert "html inject" in page
    assert "Target rank" in page


def test_benchmark_exports_public_api():
    import jlens

    assert hasattr(jlens, "BenchmarkCase")
    assert hasattr(jlens, "ReadoutCase")
    assert hasattr(jlens, "run_benchmark")
    assert hasattr(jlens, "run_readout_benchmark")
    assert hasattr(jlens, "ranked_default_cases")
    assert hasattr(jlens, "render_benchmark_html")
    assert hasattr(jlens, "render_readout_html_report")


def test_find_token_position_containing_uses_decoded_context_tokens():
    model = TinyDecoder(n_layers=4, d_model=8, seed=0)
    text = "abc def"
    ids = model.encode(text)
    decoded = model.tokenizer.decode([ids[0, 1]])

    position = find_token_position_containing(model, text, decoded)

    assert position == 1


def test_readout_case_records_best_rank_across_layers():
    model = TinyDecoder(n_layers=4, d_model=8, seed=0)
    lens = JacobianLens({1: torch.eye(8), 2: torch.eye(8)}, n_prompts=1, d_model=8)
    case = ReadoutCase(
        name="tiny readout",
        prompt=PROMPT,
        target=4,
        position=-1,
        layers=(1, 2),
        max_rank=32,
    )

    result = run_readout_case(model, lens, case)

    assert result.success
    assert result.best_rank < 32
    assert result.best_layer in {1, 2}
    assert len(result.layer_ranks) == 2


def test_readout_case_rejects_multitoken_target():
    model = TinyDecoder(n_layers=4, d_model=8, seed=0)
    lens = JacobianLens({1: torch.eye(8)}, n_prompts=1, d_model=8)

    with pytest.raises(ValueError, match="one token"):
        run_readout_case(
            model,
            lens,
            ReadoutCase(
                name="multi",
                prompt=PROMPT,
                target="ab",
                position=-1,
                layers=(1,),
            ),
        )


def test_readout_html_report_contains_case_and_best_rank():
    model = TinyDecoder(n_layers=4, d_model=8, seed=0)
    lens = JacobianLens({1: torch.eye(8)}, n_prompts=1, d_model=8)
    result = run_readout_case(
        model,
        lens,
        ReadoutCase(
            name="html readout",
            prompt=PROMPT,
            target=4,
            position=-1,
            layers=(1,),
            max_rank=32,
        ),
    )

    page = render_readout_html_report((result,), title="Viewing Report")

    assert "<!doctype html>" in page.lower()
    assert "Viewing Report" in page
    assert "html readout" in page
    assert "Best rank" in page


def test_default_viewing_cases_measure_a_completed_causal_transcript():
    import re

    from scripts.benchmark_readout import _default_cases

    class Tokenizer:
        values = {}

        def decode(self, token_ids, **kwargs):
            return "".join(self.values[token_id] for token_id in token_ids)

    class Model:
        tokenizer = Tokenizer()

        def encode(self, text, *, max_length=512):
            pieces = re.findall(r"\w+|[^\w\s]", text)[:max_length]
            self.tokenizer.values = dict(enumerate(pieces))
            return torch.tensor([list(range(len(pieces)))])

    model = Model()
    cases = _default_cases(model, max_rank=100)

    assert all(case.context_mode == "prompt+generated-response" for case in cases)
    currency = next(case for case in cases if "currency" in case.name)
    euro_position = find_token_position_containing(model, currency.prompt, "Euro")
    assert currency.position == euro_position - 1
    assert currency.prompt.endswith(" the Euro.")
