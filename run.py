#!/usr/bin/env python3
"""Lanzador unico del sistema. python run.py"""
import os, sys, subprocess, webbrowser
from pathlib import Path

MODULOS = [
    ('1', 'Model Store',               'model_store.py'),
    ('2', 'Calibracion',               'calibration.py'),
    ('3', 'Feature Store',             'feature_store.py'),
    ('4', 'Stress Test',               'stress_test.py'),
    ('5', 'Multi-Agente',              'multi_agent_system.py'),
    ('6', 'Razonamiento Causal',       'causal_reasoning.py'),
    ('7', 'Microestructura Mercado',   'market_microstructure.py'),
    ('8', 'Precios Generativos',       'generative_prices.py'),
    ('9', 'Analitica Portafolio',      'portfolio_analytics.py'),
    ('10', 'Opciones y Derivados',     'opciones_derivados.py'),
    ('11', 'CPPI / Seguro Portafolio', 'cppi.py'),
    ('12', 'Dashboard HTML',           'dashboard_app.py'),
    ('13', 'Streaming Pipeline',       'streaming_pipeline.py'),
    ('14', 'Ejecucion Algoritmica',    'ejecucion_algoritmica.py'),
    ('15', 'API REST (servidor)',      'api_rest.py'),
    ('16', 'MLOps Pipeline',           'mlops_pipeline.py'),
    ('17', 'NLU / Chat',              'interfaz_lenguaje_natural.py'),
    ('18', 'Backtest Alta Fidelidad',  'backtest_alta_fidelidad.py'),
    ('19', 'Riesgos Tiempo Real',      'riesgos_tiempo_real.py'),
    ('20', 'Conectividad Mercados',    'conectividad_mercados.py'),
    ('21', 'Persistent DB',            'persistent_db.py'),
    ('22', 'AutoML (Optuna)',          'hyperparameter_optimizer.py'),
    ('23', 'TCA (costos transaccion)', 'transaction_cost_analysis.py'),
]

SERVICIOS = [
    ('a', 'Iniciar API REST (uvicorn)', 'uvicorn api_rest:app --host 0.0.0.0 --port 8000'),
    ('b', 'Iniciar Dashboard Streamlit', 'streamlit run dashboard_interactivo.py'),
    ('c', 'Iniciar NLU interactivo',    'python interfaz_lenguaje_natural.py'),
]

TESTS = [
    ('t', 'Ejecutar todos los tests',   'python -m pytest tests/test_all.py -v'),
    ('u', 'Tests rapidos (sin optuna)', 'python -m pytest tests/test_all.py -v --ignore-glob=*Optimizador*'),
]

def limpiar():
    os.system('cls' if os.name == 'nt' else 'clear')

def menu():
    limpiar()
    print('=' * 55)
    print('  AGENTE FINANCIERO - LANZADOR UNIFICADO')
    print('=' * 55)
    print('  MODULOS:')
    for key, nombre, _ in MODULOS:
        print(f'    [{key}] {nombre}')
    print('  SERVICIOS:')
    for key, nombre, _ in SERVICIOS:
        print(f'    [{key}] {nombre}')
    print('  TESTS:')
    for key, nombre, _ in TESTS:
        print(f'    [{key}] {nombre}')
    print('  [q] Salir')
    print('=' * 55)

def ejecutar(comando, desc):
    print(f'\n--- {desc} ---')
    print(f'$ {comando}\n')
    try:
        subprocess.run(comando, shell=True, check=True)
    except KeyboardInterrupt:
        pass
    except subprocess.CalledProcessError as e:
        print(f'Error: {e}')
    input('\nPresiona Enter para volver al menu...')

def main():
    while True:
        menu()
        op = input('Selecciona una opcion: ').strip().lower()
        if op == 'q':
            print('Hasta luego.')
            break

        for key, nombre, archivo in MODULOS:
            if op == key:
                ejecutar(f'python {archivo}', nombre)
                break
        else:
            for key, nombre, comando in SERVICIOS:
                if op == key:
                    if 'uvicorn' in comando:
                        ejecutar(comando, nombre)
                    elif 'streamlit' in comando:
                        ejecutar(comando, nombre)
                    else:
                        ejecutar(comando, nombre)
                    break
            else:
                for key, nombre, comando in TESTS:
                    if op == key:
                        ejecutar(comando, nombre)
                        break
                else:
                    print('Opcion no valida.')
                    input('Presiona Enter...')

if __name__ == '__main__':
    main()
