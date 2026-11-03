#!/usr/bin/env python3
"""
build_exe.py - Compila el Agente Financiero a un solo .exe con PyInstaller.
Uso:  python build_exe.py
"""
import shutil
import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).parent
DIST = BASE / 'dist'
BUILD = BASE / 'build'
SPEC = BASE / 'agente_financiero.spec'

def color(t, c='cyan'):
    cols = {'cyan': '\x1b[36m', 'green': '\x1b[32m', 'red': '\x1b[31m', 'yellow': '\x1b[33m', 'bold': '\x1b[1m', 'end': '\x1b[0m'}
    return f"{cols.get(c, '')}{t}{cols['end']}"

def main():
    print(color('=== Build Agente Financiero .EXE ===\n', 'bold'))

    # 1. Check PyInstaller
    try:
        import PyInstaller
        print(color('  [OK] PyInstaller instalado', 'green'))
    except ImportError:
        print(color('  [!] Instalando PyInstaller...', 'yellow'))
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'pyinstaller'], check=True)

    # 2. Clean previous builds
    for d in [DIST, BUILD]:
        if d.exists():
            shutil.rmtree(d)
            print(f'  [OK] Limpiado {d.name}')

    # 3. Create .spec file
    spec_content = '''
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
'''

    with open(SPEC, 'w', encoding='utf-8') as f:
        f.write(spec_content)
    print(color('  [OK] Spec file creado', 'green'))

    # 4. Run PyInstaller
    print(color('\n=== Compilando... (toma 2-5 min) ===\n', 'bold'))
    result = subprocess.run([
        sys.executable, '-m', 'PyInstaller',
        str(SPEC),
        '--clean',
        '--noconfirm',
    ], cwd=str(BASE))

    if result.returncode != 0:
        print(color(f'\n  [!] Error compilando: {result.returncode}', 'red'))
        return

    # 5. Verify
    exe_path = DIST / 'AgenteFinanciero.exe'
    if exe_path.exists():
        size_mb = exe_path.stat().st_size / (1024 * 1024)
        print(color(f'\n  [OK] .exe creado: {exe_path}', 'green'))
        print(color(f'  [OK] Tamaño: {size_mb:.1f} MB', 'green'))
    else:
        print(color(f'\n  [!] No se encontró el .exe en {exe_path}', 'red'))

    # 6. Copy Datos folder
    datos_src = BASE / 'Datos'
    datos_dst = DIST / 'Datos'
    if datos_src.exists() and not datos_dst.exists():
        shutil.copytree(datos_src, datos_dst, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        print('  [OK] Carpeta Datos copiada')

    # 7. Copy config
    config_src = BASE / 'config'
    config_dst = DIST / 'config'
    if not config_dst.exists():
        shutil.copytree(config_src, config_dst, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        print('  [OK] Carpeta config copiada')

    print(color('\n=== COMPLETADO ===', 'bold'))
    print(f'  .exe en: {exe_path}')
    print(f'  Ejecuta: {DIST}\\AgenteFinanciero.exe')
    print(f'  Tamaño total: {size_mb:.0f} MB + Datos\n')


if __name__ == '__main__':
    main()