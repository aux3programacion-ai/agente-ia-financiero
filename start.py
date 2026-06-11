#!/usr/bin/env python3
"""
start.py - Lanzador simple del sistema.
Uso:  python start.py            (menu interactivo)
      python start.py dashboard   (lanza dashboard directo)
      python start.py api         (lanza API REST)
"""
import sys, os, subprocess, webbrowser, json
from pathlib import Path
from datetime import datetime

BASE = Path(__file__).parent
DATOS = BASE / 'Datos'

os.system('cls' if os.name == 'nt' else 'clear')

def color(t, c='cyan'):
    cols = {'cyan': '\x1b[36m', 'green': '\x1b[32m', 'red': '\x1b[31m', 'yellow': '\x1b[33m',
            'purple': '\x1b[35m', 'bold': '\x1b[1m', 'end': '\x1b[0m'}
    return f"{cols.get(c, '')}{t}{cols['end']}"

def banner():
    print(color(r'''  
        _____   _____ _   _ ____  _____ ___________ 
       / ____| |_   _| \ | |  _ \|_   _|___ /_   _|
      | |  __    | | |  \| | |_) | | |   |_ \ | |  
      | | |_ |   | | | . ` |  _ <  | |  ___) || |  
      | |__| |  _| |_| |\  | |_) |_| |_|____/ | |  
       \_____| |_____|_| \_|____/|_____|_____| |_|  
    ''', 'cyan'))
    print(color(f'              SISTEMA FINANCIERO AUTOMATIZADO v2.0', 'bold'))
    print(color(f'              {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n', 'yellow'))

def status():
    n_json = len(list(DATOS.glob('*.json'))) if DATOS.exists() else 0
    n_tca = len(list((DATOS/'tca').glob('*.txt'))) if (DATOS/'tca').exists() else 0
    n_opts = len(list((DATOS/'opciones').glob('*.json'))) if (DATOS/'opciones').exists() else 0
    n_paper = len(list((DATOS/'paper_trading').glob('*.json'))) if (DATOS/'paper_trading').exists() else 0
    print(color(f'  [DATOS] {n_json} JSONs  |  TCA: {n_tca}  |  Opciones: {n_opts}  |  Paper: {n_paper}', 'yellow'))
    try:
        branch = os.popen('git rev-parse --abbrev-ref HEAD 2>nul').read().strip()
        commit = os.popen('git log -1 --format="%h" 2>nul').read().strip()
        if branch:
            print(color(f'  [GIT] {branch} @ {commit}', 'yellow'))
    except:
        pass
    print()

def menu():
    banner()
    status()
    print(color('  LANZAR RAPIDO:\n', 'bold'))
    print(f'    {color("[1]", "cyan")}  Mercado en Vivo - precios, noticias, top picks')
    print(f'    {color("[2]", "cyan")}  Dashboard Completo (portfolio, riesgo, opciones)')
    print(f'    {color("[3]", "cyan")}  API REST (uvicorn)')
    print(f'    {color("[4]", "cyan")}  Paper Trading + Orchestrator')
    print(f'    {color("[5]", "cyan")}  Reporte TCA')
    print(f'    {color("[6]", "cyan")}  Analisis Opciones')
    print(f'    {color("[7]", "cyan")}  Menu Completo (27 modulos)')
    print(f'    {color("[q]", "red")}  Salir')
    print()

def launch(cmd, desc):
    print(color(f'\n  >> {desc}...', 'green'))
    try:
        if 'streamlit' in cmd:
            cmd = cmd.replace('streamlit run', 'python -m streamlit run')
            port = os.environ.get('PORT', '8501')
            if '$PORT' not in cmd:
                cmd += f' --server.port {port} --server.address 0.0.0.0'
            if os.environ.get('RAILWAY') or os.environ.get('RENDER'):
                print(color(f'  [Cloud] Desplegando en puerto {port}', 'yellow'))
            else:
                webbrowser.open(f'http://localhost:{port}')
        subprocess.run(cmd, shell=True, check=True)
    except KeyboardInterrupt:
        pass
    except subprocess.CalledProcessError as e:
        print(color(f'  x Error: {e}', 'red'))
    input(color('\n  Presiona Enter para volver... ', 'yellow'))

def main():
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        if arg in ('mercado', 'm'):
            launch('streamlit run mercado_tiempo_real.py', 'Mercado en Vivo')
        elif arg in ('cloud', 'c'):
            launch('python startup_cloud.py', 'Cloud Startup')
        elif arg in ('dashboard', 'd'):
            launch('streamlit run dashboard_interactivo.py', 'Dashboard Completo')
        elif arg in ('api', 'a'):
            launch('uvicorn api_rest:app --host 0.0.0.0 --port 8000', 'API REST')
        elif arg in ('paper', 'p'):
            launch('python trading_orchestrator.py', 'Paper Trading + Orchestrator')
        elif arg in ('tca', 't'):
            launch('python transaction_cost_analysis.py', 'Analisis TCA')
        elif arg in ('opciones', 'o'):
            launch('python opciones_derivados.py', 'Opciones y Derivados')
        elif arg in ('menu', 'run'):
            launch('python run.py', 'Menu Completo')
        else:
            print(color('  Uso: python start.py [mercado|dashboard|api|paper|tca|opciones|menu]', 'yellow'))
        return

    while True:
        os.system('cls' if os.name == 'nt' else 'clear')
        menu()
        op = input(color('  Selecciona: ', 'bold')).strip().lower()
        actions = {
            '1': ('streamlit run mercado_tiempo_real.py', 'Mercado en Vivo'),
            '2': ('streamlit run dashboard_interactivo.py', 'Dashboard Completo'),
            '3': ('uvicorn api_rest:app --host 0.0.0.0 --port 8000', 'API REST'),
            '4': ('python trading_orchestrator.py', 'Paper Trading + Orchestrator'),
            '5': ('python transaction_cost_analysis.py', 'Analisis TCA'),
            '6': ('python opciones_derivados.py', 'Analisis Opciones'),
            '7': ('python run.py', 'Menu Completo (27 modulos)'),
        }
        if op == 'q':
            print(color('\n  Hasta luego!\n', 'green'))
            break
        elif op in actions:
            launch(*actions[op])
        else:
            input(color('  Opcion no valida. Enter... ', 'red'))

if __name__ == '__main__':
    main()
