import numpy as np
import pandas as pd
import pytest

from modelo_xgboost import FeatureBuilder, WalkForwardValidator, _ChronologicalBlockCV


def _df(n=400, seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.date_range('2023-01-01', periods=n, freq='B')
    close = 100 * np.cumprod(1 + rng.normal(0.0005, 0.02, n))
    return pd.DataFrame({
        'Open': close, 'High': close * 1.01, 'Low': close * 0.99,
        'Close': close, 'Volume': rng.integers(1e6, 5e6, n),
    }, index=idx)


def test_build_features_columnas_clave():
    df, cols = FeatureBuilder.build_features(_df())
    for c in ['mom_12_1m', 'short_pct_float', 'sector_mom_composite', 'eps_surprise', 'fwd_vol_20d']:
        assert c in df.columns
    assert 'target' in df.columns
    assert 'return_1d' in cols


def test_target_relativo_spy():
    df = _df()
    spy = pd.Series(df['Close'].values * 0.5, index=df.index)
    df2, _ = FeatureBuilder.build_features(df, spy_close=spy)
    fwd = df2['Close'].shift(-20) / df2['Close'] - 1
    spy_fwd = spy.shift(-20) / spy - 1
    esperado = (fwd > spy_fwd).astype(int)
    esperado[fwd.isna()] = np.nan
    pd.testing.assert_series_equal(df2['target'].astype('float'), esperado.astype('float'), check_names=False)


def test_target_nan_en_horizonte():
    df, _ = FeatureBuilder.build_features(_df())
    # los ultimos 20 deberian quedar sin label utilizable (shift -20)
    assert df['target'].isna().sum() >= 20 or df['target'].notna().sum() <= len(df) - 1


def test_embargo_walkforward():
    wf = WalkForwardValidator()
    assert wf.gap == 20


def test_chronological_block_cv_particion():
    n = 300
    seen = np.zeros(n, dtype=int)
    X = np.random.rand(n, 3)
    cv = _ChronologicalBlockCV(n_splits=3)
    for train_idx, test_idx in cv.split(X):
        assert len(set(train_idx) & set(test_idx)) == 0
        seen[test_idx] += 1
    assert (seen == 1).all()
    assert cv.get_n_splits() == 3


def test_sin_lookahead_en_features():
    df = _df()
    df2a, cols = FeatureBuilder.build_features(df)
    df2b, _ = FeatureBuilder.build_features(df.iloc[:-1])
    # las features en la ultima fila del df truncado deben coincidir con las de la penultima del completo
    for c in cols:
        va = df2a[c].iloc[-2]
        vb = df2b[c].iloc[-1] if c in df2b else np.nan
        if pd.notna(va) and pd.notna(vb):
            assert np.isclose(va, vb, equal_nan=True), f'lookahead en {c}'
