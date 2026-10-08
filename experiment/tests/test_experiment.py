"""experiment/ tooling vs. the kit, on the synthetic schema-2 fixture.

Run from the repo root:  python -m pytest experiment/tests -q
"""

from __future__ import annotations

import contextlib
import datetime
import io
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from f522kit.calendar import Anchor
from f522kit.data import Dataset
from f522kit.driver import run
from f522kit.fixture import build_fixture
from f522kit.scoring import score
from f522kit.window import TrainWindow

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from experiment import run as xrun, split as xsplit, table as xtable  # noqa: E402
from experiment.checks import stride_test  # noqa: E402
from models.ridge import MyModel as Ridge  # noqa: E402

FIXTURE_VAL_AFTER = datetime.date(2023, 2, 1)  # the fixture spans 2023-01..03
WEEKS = 4


def quiet(fn, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


@pytest.fixture(scope="module", autouse=True)
def fixture_val_after():
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(xsplit, "VAL_AFTER", FIXTURE_VAL_AFTER)
        yield


@pytest.fixture(scope="module")
def data_root(tmp_path_factory):
    return build_fixture(tmp_path_factory.mktemp("fx") / "data", schema_version=2)


def build(data_root, out, stride):
    quiet(xsplit.build_split, data_root, out, train_weeks=WEEKS, minute_stride=stride,
          workers=2)
    return xsplit.ExperimentSplit(out)


@pytest.fixture(scope="module")
def split1(data_root, tmp_path_factory):
    return build(data_root, tmp_path_factory.mktemp("s") / "stride1", 1)


@pytest.fixture(scope="module")
def split3(data_root, tmp_path_factory):
    return build(data_root, tmp_path_factory.mktemp("s") / "stride3", 3)


def kit_window(data_root):
    ds = Dataset(data_root)
    anchor, _, groups = xsplit.choose_anchor(ds)
    trunc = Anchor(anchor.key, anchor.position, anchor.training_keys[-WEEKS:])
    return TrainWindow(ds, trunc, trunc.training_dates(groups))


# -- split ---------------------------------------------------------------------

def test_stride1_train_rows_equal_kit_trainwindow(data_root, split1):
    blocks = list(kit_window(data_root).iter_dates())
    X, y, w, _ = split1.train()
    np.testing.assert_array_equal(X, np.concatenate([b.x for b in blocks]))
    np.testing.assert_array_equal(y, np.concatenate([b.y for b in blocks]))
    np.testing.assert_array_equal(w, np.concatenate([b.w_train for b in blocks]))


def test_val_is_the_anchors_scored_dates(data_root, split1):
    _, val_dates, _ = xsplit.choose_anchor(Dataset(data_root))
    assert split1.val_dates == val_dates
    assert min(split1.val_dates) > max(split1.train_dates)
    _, _, w, _ = split1.val()
    assert np.all(w > 0)


def test_stride3_is_a_subset_of_stride1(split1, split3):
    m3 = np.load(split3.root / "train_minute.npy")
    assert np.all(m3 % 3 == 0)
    key = lambda s: set(zip(np.load(s.root / "train_date.npy").tolist(),  # noqa: E731
                            np.load(s.root / "train_row.npy").tolist()))
    assert key(split3) < key(split1)
    assert split3.meta["val"] == split1.meta["val"]  # val is never strided


def test_split_refuses_overwrite_unless_forced(data_root, split3):
    with pytest.raises(FileExistsError):
        build(data_root, split3.root, 3)
    quiet(xsplit.build_split, data_root, split3.root, train_weeks=WEEKS, minute_stride=3,
          workers=2, force=True)
    assert xsplit.ExperimentSplit(split3.root).meta["minute_stride"] == 3


# -- models ----------------------------------------------------------------------

def test_ridge_fit_arrays_equals_kit_fit(data_root, split1):
    kit = Ridge()
    kit.fit(kit_window(data_root))
    arrays = Ridge()
    arrays.fit_arrays(*split1.train())
    np.testing.assert_allclose(arrays.beta, kit.beta, rtol=1e-9, atol=1e-15)


@pytest.mark.parametrize("model,config", [
    ("ridge", {}),
    ("lgbm", {"num_boost_round": 50, "early_stopping_rounds": 10, "es_days": 5,
              "min_data_in_leaf": 20, "num_leaves": 7, "threads": 1}),
    ("mlp", {"hidden": [16], "max_epochs": 30, "patience": 5, "es_days": 5,
             "batch_size": 64, "threads": 1}),  # ~1k rows: needs many small steps
])
def test_three_models_run_and_find_signal(split3, tmp_path, model, config):
    rec = quiet(xrun.run_one, split3.root, model, config, model, tmp_path)
    assert rec["report"]["windows"]["val"]["COR_mean"] > 0.2  # planted linear signal
    assert rec["timing_s"]["fit"] >= 0 and rec["peak_rss_gb"]["final"] > 0


def test_equal_data_settings_give_identical_fit_rows(split3, tmp_path):
    """Same max_rows / es_days / seed -> all three models fit the same rows."""
    common = {"max_rows": 400, "es_days": 5, "seed": 3}
    configs = {
        "ridge": common,
        "lgbm": {**common, "num_boost_round": 5, "min_data_in_leaf": 20, "num_leaves": 7,
                 "threads": 1},
        "mlp": {**common, "hidden": [8], "max_epochs": 1, "threads": 1},
    }
    shas = {}
    for model, cfg in configs.items():
        rec = quiet(xrun.run_one, split3.root, model, cfg, model, tmp_path)
        assert rec["fit_info"]["n_fit_rows"] == 400
        shas[model] = rec["fit_info"]["fit_rows_sha"]
    assert len(set(shas.values())) == 1, shas
    # Without max_rows, ridge with es_days drops exactly the holdout + embargo rows,
    # matching the MLP's all-rows fit set.
    r = quiet(xrun.run_one, split3.root, "ridge", {"es_days": 5}, "r_es", tmp_path)
    m = quiet(xrun.run_one, split3.root, "mlp", {"es_days": 5, "hidden": [8], "max_epochs": 1,
                                                  "threads": 1}, "m_es", tmp_path)
    assert r["fit_info"]["fit_rows_sha"] == m["fit_info"]["fit_rows_sha"]
    assert r["fit_info"]["n_fit_rows"] < split3.meta["train"]["n_rows"]


# -- run -----------------------------------------------------------------------

def test_run_writes_kit_format_preds_scored_by_kit(data_root, split3, tmp_path):
    rec = quiet(xrun.run_one, split3.root, "ridge", {}, "r", tmp_path)
    ds = Dataset(data_root)
    _, _, _, d = split3.val()
    rows = split3.val_rows()
    for i, date in enumerate(split3.val_dates):
        grid = np.load(tmp_path / "r" / "preds" / f"{date.isoformat()}.npy")
        assert grid.shape == (ds.time_rows, ds.shard(date).n_symbols)
        skipped = np.setdiff1d(np.arange(grid.size), rows[d == i])
        assert np.all(grid.reshape(-1)[skipped] == 0)
    window = {"val": (split3.val_dates[0], split3.val_dates[-1])}
    assert rec["report"]["windows"] == score(ds, tmp_path / "r", windows=window)["windows"]
    saved = json.loads((tmp_path / "r" / "record.json").read_text())
    assert len(saved["report"]["daily_APS_bps"]) == len(split3.val_dates)


def test_run_refuses_existing_name_and_missing_baseline(split3, tmp_path):
    quiet(xrun.run_one, split3.root, "ridge", {}, "r", tmp_path)
    with pytest.raises(FileExistsError):
        quiet(xrun.run_one, split3.root, "ridge", {}, "r", tmp_path)
    with pytest.raises(FileNotFoundError):  # raised before any fitting
        quiet(xrun.run_one, split3.root, "ridge", {}, "r2", tmp_path,
              baseline=tmp_path / "not_there")
    assert not (tmp_path / "r2").exists()


def test_experiment_list(split3, tmp_path):
    lst = tmp_path / "exp.txt"
    lst.write_text("# comment\n\nr_a  ridge  {}\nr_b  ridge  {\"lam\": 10.0}  # trailing\n")
    assert xrun.read_list(lst) == [("r_a", "ridge", {}), ("r_b", "ridge", {"lam": 10.0})]
    quiet(xrun.main, ["--split", str(split3.root), "--out", str(tmp_path / "res"),
                      "--list", str(lst), "--index", "2", "--suffix", "_c1"])
    rec = json.loads((tmp_path / "res" / "r_b_c1" / "record.json").read_text())
    assert rec["config"] == {"lam": 10.0}
    lst.write_text("x ridge {}\nx ridge {}\n")
    with pytest.raises(SystemExit):
        xrun.read_list(lst)


# -- table -------------------------------------------------------------------------

def test_table_pairs_runs_and_includes_kit_run(data_root, split3, tmp_path):
    ds = Dataset(data_root)
    quiet(run, ds, f"{REPO}/models/ridge.py:MyModel", tmp_path / "kitrun", stride="4w",
          row_budget=10_000, seed=0, start=ds.dates("visible")[0], verbose=False)
    res = tmp_path / "res"
    quiet(xrun.run_one, split3.root, "ridge", {}, "ridge", res)
    quiet(xrun.run_one, split3.root, "ridge", {"lam": 10.0}, "ridge_lam10", res)
    rows, md = xtable.build_table(split3.root, res, baseline=res / "ridge",
                                  extra=[tmp_path / "kitrun"])
    by = {r["name"]: r for r in rows}
    assert set(by) == {"ridge", "ridge_lam10", "kitrun"}
    assert by["ridge"]["dAPS_bps"] == 0.0
    assert by["kitrun"]["fit_s"] is None and "kitrun" in md


# -- checks/stride_test -------------------------------------------------------------

def test_stride_test_matches_split_plus_ridge(data_root, split1, split3, tmp_path):
    out = tmp_path / "stride.json"
    quiet(stride_test.main, ["--data", str(data_root), "--train-weeks", str(WEEKS),
                             "--strides", "3,1", "--readers", "2", "--out", str(out)])
    res = json.loads(out.read_text())["results"]
    for stride, sp in ((3, split3), (1, split1)):
        rec = quiet(xrun.run_one, sp.root, "ridge", {}, f"r{stride}", tmp_path / "res")
        assert res[str(stride)]["train_rows"] == sp.meta["train"]["n_rows"]
        np.testing.assert_allclose(res[str(stride)]["APS_mean_bps"],
                                   rec["report"]["windows"]["val"]["APS_mean_bps"], rtol=1e-6)


def test_config_table_combines_seeds(split3, tmp_path):
    res = tmp_path / "res"
    quiet(xrun.run_one, split3.root, "ridge", {"es_days": 5}, "ridge_es", res)
    quiet(xrun.run_one, split3.root, "ridge", {"lam": 10.0}, "ridge_lam10", res)
    mlp = {"es_days": 5, "hidden": [8], "max_epochs": 2, "threads": 1}
    for s in (0, 1, 2):
        quiet(xrun.run_one, split3.root, "mlp", {**mlp, "seed": s}, f"mlp_s{s}", res)
    quiet(xrun.run_one, split3.root, "mlp", {**mlp, "lr": 3e-4}, "mlp_lr", res)

    rows, md = xtable.build_config_table(res, baseline="ridge_es")
    by = {r["runs"]: r for r in rows}
    assert set(by) == {"ridge_es", "ridge_lam10", "mlp_s0, mlp_s1, mlp_s2", "mlp_lr"}
    assert rows[0]["note"] == "baseline" and rows[0]["dAPS_bps"] is None
    seeds = by["mlp_s0, mlp_s1, mlp_s2"]
    assert seeds["seeds"] == 3 and seeds["note"] == "" and seeds["rows"] == "same"
    aps = [json.loads((res / f"mlp_s{s}" / "record.json").read_text())
           ["report"]["windows"]["val"]["APS_mean_bps"] for s in (0, 1, 2)]
    np.testing.assert_allclose(seeds["APS_bps"], np.mean(aps))
    assert by["mlp_lr"]["note"].startswith("⚠")    # one seed of a random model
    assert by["ridge_lam10"]["note"] == ""          # ridge is deterministic
    # paired columns: seed-averaged daily APS minus the baseline's, day by day
    base = json.loads((res / "ridge_es" / "record.json").read_text())["report"]["daily_APS_bps"]
    daily = np.mean([json.loads((res / f"mlp_s{s}" / "record.json").read_text())
                     ["report"]["daily_APS_bps"] for s in (0, 1, 2)], axis=0)
    np.testing.assert_allclose(seeds["dAPS_bps"], np.mean(daily - np.array(base)))
    assert "mlp_s0, mlp_s1, mlp_s2" in md


def test_resolved_config_separates_changed_defaults(split3, tmp_path):
    """Same typed settings but different effective settings -> different rows."""
    res = tmp_path / "res"
    quiet(xrun.run_one, split3.root, "ridge", {}, "r_default", res)
    quiet(xrun.run_one, split3.root, "ridge", {"es_days": 5}, "r_es", res)
    rec = json.loads((res / "r_default" / "record.json").read_text())
    assert rec["resolved_config"]["lam"] == 0.01 and rec["resolved_config"]["es_days"] == 0
    # Simulate an old record whose model default differed: same typed config {},
    # different resolved settings. They must not be averaged together.
    old = dict(rec, name="r_old", resolved_config={**rec["resolved_config"], "lam": 1.0})
    (res / "r_old").mkdir()
    (res / "r_old" / "record.json").write_text(json.dumps(old))
    rows, _ = xtable.build_config_table(res, baseline="r_es")
    assert {r["runs"] for r in rows} == {"r_default", "r_old", "r_es"}


def test_config_table_resolves_records_without_settings(split3, tmp_path):
    """A record written by older code (no resolved_config) groups with new ones."""
    res = tmp_path / "res"
    quiet(xrun.run_one, split3.root, "ridge", {"es_days": 5}, "base", res)
    quiet(xrun.run_one, split3.root, "ridge", {"lam": 3.0}, "new", res)
    old = json.loads((res / "new" / "record.json").read_text())
    old.pop("resolved_config")
    old["name"] = "legacy"
    (res / "legacy").mkdir()
    (res / "legacy" / "record.json").write_text(json.dumps(old))
    rows, _ = xtable.build_config_table(res, baseline="base")
    assert {r["runs"] for r in rows} == {"base", "legacy, new"}
