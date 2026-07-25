from __future__ import annotations

from pathlib import Path

from evaluation import expand_evaluation_runs, load_evaluation_set
from public_cli import main as public_main

REPO_ROOT = Path(__file__).resolve().parents[1]
README_PATH = REPO_ROOT / "README.md"
OUTPUT_GUIDE_PATH = REPO_ROOT / "docs/output-guide.md"


def test_repo_guides_point_to_public_cli() -> None:
    for path in (
        "README.md",
        "AGENTS.md",
        "docs/improvements.md",
    ):
        text = Path(path).read_text(encoding="utf-8")
        assert "monte-carlo simulate" in text
        assert "python3 -m pip install -e ." in text


def test_contributor_guides_describe_public_cli_split() -> None:
    for path in ("AGENTS.md",):
        text = Path(path).read_text(encoding="utf-8")
        assert "public_cli.py" in text


def test_contributor_guides_name_evaluation_contract_and_reference_sets() -> None:
    text = Path("AGENTS.md").read_text(encoding="utf-8")
    assert "evaluation.py" in text
    assert "evaluation_sets/" in text


def test_improvements_doc_avoids_retired_command_examples() -> None:
    text = Path("docs/improvements.md").read_text(encoding="utf-8")
    assert "python MonteCarlo.py --ticker" not in text
    assert "python cli.py --ticker" not in text
    assert "pip install -r requirements.txt" not in text


def test_readme_offline_backtest_example_uses_sample_data_friendly_window() -> None:
    text = README_PATH.read_text(encoding="utf-8")
    assert "sample_data" in text
    assert "backtest" in text.lower()


def test_readme_distinguishes_cli_commands_from_browser_entrypoint() -> None:
    text = README_PATH.read_text(encoding="utf-8")
    assert "## CLI" in text
    assert "npm run dev" in text
    assert "monte-carlo simulate" in text


def test_readme_explains_how_to_read_results_and_saved_backtest_outputs() -> None:
    text = README_PATH.read_text(encoding="utf-8")
    assert "provenance" in text.lower()
    assert "docs/deploy.md" in text


def test_readme_links_output_guide_and_stays_operator_focused() -> None:
    text = README_PATH.read_text(encoding="utf-8")
    assert "docs/deploy.md" in text
    assert "monte-carlo simulate" in text
    assert "monte-carlo backtest" in text


def test_readme_documents_evaluation_gate_before_capital_is_risked() -> None:
    text = README_PATH.read_text(encoding="utf-8")
    assert (
        "monte-carlo evaluate evaluation_sets/sample-stability.json "
        "--output results/evaluation"
    ) in text
    assert "before capital is risked" in text


def test_output_guide_names_key_saved_artifacts() -> None:
    text = OUTPUT_GUIDE_PATH.read_text(encoding="utf-8")
    for artifact in (
        "action_plan.md",
        "report.json",
        "rankings.csv",
        "allocations.csv",
        "summaries.csv",
        "backtest_summary.csv",
        "rebalance_log.csv",
        "equity_curve.csv",
        "equity_curve.png",
        "price_sources.json",
    ):
        assert artifact in text
    assert "execution_plan.csv" not in text
    assert "simulations.csv.gz" not in text


def test_output_guide_explains_evaluation_artifacts() -> None:
    text = OUTPUT_GUIDE_PATH.read_text(encoding="utf-8")
    assert "scorecard.md" in text
    assert "runs.csv" in text
    assert "manifest hash" in text
    assert "normalized matrix" in text
    assert "source reliability" in text


def test_reference_evaluation_set_expands_to_bundled_sample_data() -> None:
    evaluation_set = load_evaluation_set("evaluation_sets/sample-stability.json")

    assert len(expand_evaluation_runs(evaluation_set)) == 6
    assert evaluation_set.sources[0].data_path == (REPO_ROOT / "sample_data").resolve()


def test_readme_offline_examples_run(capsys) -> None:
    sample_data = str(REPO_ROOT / "sample_data")

    simulate_exit = public_main(
        ["simulate", "AAPL", "MSFT", "--source", "offline", "--data-path", sample_data]
    )
    simulate_output = capsys.readouterr().out
    assert simulate_exit == 0
    assert "Stance:" in simulate_output
    assert "Top idea: AAPL" in simulate_output

    backtest_exit = public_main(
        [
            "backtest",
            "AAPL",
            "MSFT",
            "--source",
            "offline",
            "--data-path",
            sample_data,
            "--lookback",
            "5",
            "--hold",
            "3",
            "--rebalance",
            "3",
            "--top",
            "1",
            "--scenarios",
            "10",
        ]
    )
    backtest_output = capsys.readouterr().out
    assert backtest_exit == 0
    assert "Strategy return:" in backtest_output
