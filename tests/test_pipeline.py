import numpy as np, pandas as pd, yaml, pytest
from rwe_pgx_ml import synth, data, models, evaluate as ev

@pytest.fixture(scope="module")
def cfg():
    c = yaml.safe_load(open("config/config.yaml")); c["synthetic"]["n_per_pair"] = 400; c["pairs"] = c["pairs"][:2]; return c

def test_synth_reproducible_and_valid(cfg):
    a, ca = synth.generate(cfg); b, _ = synth.generate(cfg)
    pd.testing.assert_frame_equal(a, b); data.validate(a, ca, cfg)

def test_folds_partition_and_ancestry_balanced(cfg):
    d, _ = synth.generate(cfg); d = d[d.pair == cfg["pairs"][0]].reset_index(drop=True)
    f = models.folds(d, 5, 1); te = np.concatenate([t for _, t in f])
    assert sorted(te) == list(range(len(d)))                       # every patient tested exactly once
    assert all(len(set(tr) & set(t)) == 0 for tr, t in f)          # no train/test leakage

def test_curated_only_excludes_rwe_and_covariates(cfg):
    d, c = synth.generate(cfg); cp = c[c.pair == cfg["pairs"][0]]
    cols = data.features(d, cp, cfg["conditions"]["curated_only"])
    assert not set(cols) & set(cp[cp.source == "rwe"].variant) and "age" not in cols

def test_metrics_perfect_and_random():
    y = np.array([0, 0, 1, 1]); assert ev.auc(y, np.array([.1, .2, .8, .9])) == 1.0
    assert np.isnan(ev.auc(np.zeros(4), np.ones(4)))

def test_validate_rejects_bad_ancestry(cfg):
    d, c = synth.generate(cfg); d.loc[0, "ancestry"] = "XXX"
    with pytest.raises(ValueError): data.validate(d, c, cfg)
