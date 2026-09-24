import datetime

import numpy as np
import pytest

from f522kit import contract
from f522kit.calendar import build_schedule
from f522kit.data import DataError, Dataset
from f522kit.fixture import build_fixture
from f522kit.window import TrainWindow


@pytest.fixture(scope="module")
def fixture_root(tmp_path_factory):
    return build_fixture(tmp_path_factory.mktemp("fixture") / "data")


@pytest.fixture(scope="module")
def dataset(fixture_root):
    return Dataset(fixture_root)


def test_dataset_splits(dataset):
    visible = dataset.dates("visible")
    hidden = dataset.dates("hidden")
    assert len(visible) + len(hidden) == len(dataset.dates("all"))
    assert max(visible) < dataset.hidden_start <= min(hidden)


def test_visible_has_labels_hidden_does_not(dataset):
    v = dataset.shard(dataset.dates("visible")[0])
    h = dataset.shard(dataset.dates("hidden")[0])
    assert v.has_labels
    assert not h.has_labels


def test_deep_verify_passes_and_detects_corruption(tmp_path):
    root = build_fixture(tmp_path / "data")
    dataset = Dataset(root)
    dataset.verify(deep=True)

    # Corrupt one byte of one array; deep verify must fail.
    victim = root / "dates" / dataset.dates("visible")[3].isoformat() / "X.npy"
    payload = bytearray(victim.read_bytes())
    payload[-1] ^= 0xFF
    victim.write_bytes(bytes(payload))
    with pytest.raises(DataError, match="sha256 mismatch"):
        dataset.verify(deep=True)
    # Structural verify alone does not catch it.
    dataset.verify(deep=False)


def test_flat_rows_zero_copy_shapes(dataset):
    shard = dataset.shard(dataset.dates("visible")[0])
    x2d, y, w = shard.flat_rows()
    assert x2d.shape == (dataset.time_rows * shard.n_symbols, dataset.n_features)
    assert y.shape == w.shape == (dataset.time_rows * shard.n_symbols,)


def _window(dataset):
    visible = dataset.dates("visible")
    schedule = build_schedule(visible, stride_weeks=1)
    anchor = schedule.anchors[-1]
    return TrainWindow(dataset, anchor, anchor.training_dates(schedule.groups))


def test_window_never_contains_anchor_or_future(dataset):
    window = _window(dataset)
    assert window.dates
    assert all(d < window.anchor_date for d in window.dates)


def test_window_blocks_drop_zero_weight_rows(dataset):
    window = _window(dataset)
    for block in window.iter_dates():
        assert np.all(block.w_train > 0)
        assert block.x.shape[0] == block.y.shape[0] == block.w_train.shape[0]


def test_sample_deterministic_and_stratified(dataset):
    window = _window(dataset)
    x1, y1, w1 = window.sample(500, seed=42)
    x2, y2, w2 = window.sample(500, seed=42)
    np.testing.assert_array_equal(x1, x2)
    np.testing.assert_array_equal(y1, y2)
    np.testing.assert_array_equal(w1, w2)
    assert x1.shape == (500, dataset.n_features)

    x3, _, _ = window.sample(500, seed=43)
    assert not np.array_equal(x1, x3)


def test_sample_respects_row_budget(dataset):
    visible = dataset.dates("visible")
    schedule = build_schedule(visible, stride_weeks=1)
    anchor = schedule.anchors[-1]
    window = TrainWindow(
        dataset, anchor, anchor.training_dates(schedule.groups), row_budget=100
    )
    window.sample(100, seed=0)
    with pytest.raises(ValueError, match="exceeds the frozen training-row budget"):
        window.sample(101, seed=0)


def test_sample_decay_applied(dataset):
    window = _window(dataset)
    x, y, w_train = window.sample(200, seed=0)
    # Decay factors are <= 1 and the raw fixture weights are in [0.5, 2.0],
    # so decayed training weights stay within (0, 2.0].
    assert np.all(w_train > 0)
    assert np.all(w_train <= 2.0 + 1e-12)
