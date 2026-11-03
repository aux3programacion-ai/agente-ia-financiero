
# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path

a = Analysis(
    ['launcher_tk.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=[
        'streamlit', 'plotly', 'pandas', 'numpy', 'yfinance',
        'xgboost', 'sklearn', 'fastapi', 'uvicorn', 'apscheduler',
        'structlog', 'yaml', 'dotenv', 'requests',
        'tkinter', 'PIL',
        'sklearn.ensemble', 'sklearn.model_selection',
        'sklearn.calibration', 'sklearn.isotonic',
        'sklearn.linear_model', 'sklearn.tree',
        'scipy.stats', 'scipy.spatial',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'matplotlib', 'tensorflow', 'torch', 'transformers',
        'lightgbm', 'catboost', 'optuna', 'shap',
        'notebook', 'jupyter', 'ipython',
        'selenium', 'playwright', 'kivy',
    ],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='AgenteFinanciero',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='icon.ico' if Path('icon.ico').exists() else None,
)
