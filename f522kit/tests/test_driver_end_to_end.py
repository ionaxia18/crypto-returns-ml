"""End-to-end: fixture -> run -> score, with a planted signal to recover."""

import json

import numpy as np
import pytest

from f522kit.data import Dataset
from f522kit.driver import run
from f522kit.fixture import build_fixture
from f522kit.scoring import ScoringError, score

RIDGE_MODEL = """
import numpy as np

class TinyRidge:
    def fit(self, train):
        X, y, w = train.sample(20000, seed=0)
        Xw = X * w[:, None]
        xx = Xw.T @ X + 1e-6 * np.eye(X.shape[1])
        xy = Xw.T @ y
        self.beta = np.linalg.solve(xx, xy)

    def predict(self, X):
        return X @ self.beta
"""


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("e2e")
    root = build_fixture(tmp / "data")
    dataset = Dataset(root)
    model_path = tmp / "model.py"
    model_path.write_text(RIDGE_MODEL, encoding="utf-8")
    out = tmp / "run"
    manifest = run(
        dataset,
        model_spec=f"{model_path}:TinyRidge",
        out_dir=out,
        stride="1w",
        row_budget=20000,
        seed=0,
        start=dataset.dates("visible")[0],
        verbose=False,
    )
    return dataset, out, manifest, tmp


def test_manifest_covers_all_scored_visible_dates(pipeline):
    dataset, out, manifest, _ = pipeline
    preds = sorted((out / "preds").glob("*.npy"))
    assert len(preds) == len(manifest["predictions"])
    assert manifest["dataset_id"] == dataset.dataset_id
    for date, entry in manifest["predictions"].items():
        assert entry["anchor"] < date  # each anchor strictly precedes its date


def test_planted_signal_recovered(pipeline):
    dataset, out, _, _ = pipeline
    first = dataset.dates("visible")[0]
    last = dataset.dates("visible")[-1]
    report = score(dataset, preds_dir=out, windows={"all": (first, last)})
    cor = report["windows"]["all"]["COR_mean"]
    assert cor > 0.2, f"planted linear signal should be recoverable, got COR {cor}"
    assert report["guardrails"]["all"]["clip_rate"] < 0.01


def test_run_is_resumable_and_deterministic(pipeline):
    dataset, out, manifest, tmp = pipeline
    model_path = tmp / "model.py"
    before = {
        k: v["sha256"] for k, v in manifest["predictions"].items()
    }
    manifest2 = run(
        dataset,
        model_spec=f"{model_path}:TinyRidge",
        out_dir=out,
        stride="1w",
        row_budget=20000,
        seed=0,
        start=dataset.dates("visible")[0],
        verbose=False,
    )
    after = {k: v["sha256"] for k, v in manifest2["predictions"].items()}
    assert before == after


def test_settings_mismatch_refuses_to_reuse_out_dir(pipeline):
    dataset, out, _, tmp = pipeline
    model_path = tmp / "model.py"
    with pytest.raises(RuntimeError, match="different settings"):
        run(
            dataset,
            model_spec=f"{model_path}:TinyRidge",
            out_dir=out,
            stride="1w",
            row_budget=20000,
            seed=1,  # changed seed, same out dir
            start=dataset.dates("visible")[0],
            verbose=False,
        )


def test_model_code_change_refuses_stale_resume(pipeline):
    dataset, out, _, tmp = pipeline
    model_path = tmp / "model.py"
    original = model_path.read_text(encoding="utf-8")
    try:
        model_path.write_text(original + "\n# edited\n", encoding="utf-8")
        with pytest.raises(RuntimeError, match="model file changed"):
            run(
                dataset,
                model_spec=f"{model_path}:TinyRidge",
                out_dir=out,
                stride="1w",
                row_budget=20000,
                seed=0,
                start=dataset.dates("visible")[0],
                verbose=False,
            )
    finally:
        model_path.write_text(original, encoding="utf-8")


def test_row_budget_is_required(pipeline, tmp_path):
    dataset, _, _, tmp = pipeline
    model_path = tmp / "model.py"
    with pytest.raises(ValueError, match="row_budget is required"):
        run(
            dataset,
            model_spec=f"{model_path}:TinyRidge",
            out_dir=tmp_path / "nobudget",
            stride="1w",
            row_budget=None,
            seed=0,
            verbose=False,
        )


def test_stray_npy_in_preds_dir_is_ignored(pipeline):
    dataset, out, _, _ = pipeline
    stray = out / "preds" / "debug_betas.npy"
    np.save(stray, np.zeros(3), allow_pickle=False)
    try:
        first = dataset.dates("visible")[0]
        last = dataset.dates("visible")[-1]
        report = score(dataset, preds_dir=out, windows={"all": (first, last)})
        assert report["windows"]["all"]["n_dates"] > 0
    finally:
        stray.unlink()


def test_empty_window_fails_loud_or_reports(pipeline):
    dataset, out, _, _ = pipeline
    import datetime

    empty = (datetime.date(2019, 1, 1), datetime.date(2019, 1, 31))
    with pytest.raises(ScoringError, match="no predictable"):
        score(dataset, preds_dir=out, windows={"empty": empty})
    report = score(
        dataset,
        preds_dir=out,
        windows={"empty": empty},
        require_all_dates=False,
    )
    assert report["guardrails"]["empty"]["empty_window"] is True
    assert "empty" not in report["windows"]


def test_missing_dates_fail_scoring(pipeline):
    dataset, out, _, _ = pipeline
    scored = sorted((out / "preds").glob("*.npy"))
    victim = scored[len(scored) // 2]
    payload = victim.read_bytes()
    try:
        victim.unlink()
        first = dataset.dates("visible")[0]
        last = dataset.dates("visible")[-1]
        with pytest.raises(ScoringError, match="missing predictions"):
            score(dataset, preds_dir=out, windows={"all": (first, last)})
    finally:
        victim.write_bytes(payload)


def test_baseline_paired_diff(pipeline, tmp_path):
    dataset, out, _, tmp = pipeline
    # Baseline: sign-flipped copy of the same predictions (strictly worse).
    base = tmp_path / "baseline" / "preds"
    base.mkdir(parents=True)
    for path in (out / "preds").glob("*.npy"):
        np.save(base / path.name, -np.load(path), allow_pickle=False)
    first = dataset.dates("visible")[0]
    last = dataset.dates("visible")[-1]
    report = score(
        dataset,
        preds_dir=out,
        windows={"all": (first, last)},
        baseline_dir=base.parent,
    )
    diff = report["vs_baseline"]["all"]
    assert diff["n_dates"] == report["windows"]["all"]["n_dates"]
    assert diff["dAPS_mean_bps"] > 0
    assert diff["win_rate"] > 0.9
