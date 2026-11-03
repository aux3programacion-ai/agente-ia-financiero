#!/usr/bin/env python3
"""
launcher.py - Punto de entrada único para el .exe.
Reemplaza start.py y run.py: menú gráfico Tkinter que lanza todo.
"""
import os
import subprocess
import sys
import threading
import webbrowser
from datetime import datetime
from pathlib import Path

# Determinar ruta base: si es .exe, usar el directorio del ejecutable
if getattr(sys, 'frozen', False):
    BASE = Path(sys.executable).parent
else:
    BASE = Path(__file__).parent

os.chdir(str(BASE))
os.environ['GITHUB_WORKSPACE'] = str(BASE)

try:
    import tkinter as tk
    from tkinter import messagebox, ttk
    HAS_TK = True
except ImportError:
    HAS_TK = False


def color(t, c='cyan'):
    cols = {'cyan': '\x1b[36m', 'green': '\x1b[32m', 'red': '\x1b[31m', 'yellow': '\x1b[33m',
            'bold': '\x1b[1m', 'end': '\x1b[0m'}
    return f"{cols.get(c, '')}{t}{cols['end']}"


MODULOS = [
    ('Dashboard',       'streamlit run dashboard_interactivo.py',       '📊'),
    ('Mercado en Vivo', 'streamlit run mercado_tiempo_real.py',         '📈'),
    ('API REST',        'uvicorn api_rest:app --host 0.0.0.0 --port 8000', '🌐'),
    ('Paper Trading',   'python trading_orchestrator.py',              '💹'),
    ('Pipeline Completo','python circuito_completo.py --quick',         '⚙️'),
    ('Backtest',        'python backtest_alta_fidelidad.py',           '📉'),
    ('Análisis IA',     'python analisis_ia.py',                       '🤖'),
    ('Portafolio',      'python optimizacion_portafolio.py',           '💰'),
    ('Opciones',        'python opciones_derivados.py',                '🔧'),
    ('Riesgos',         'python riesgos_tiempo_real.py',              '⚠️'),
    ('TCA',             'python transaction_cost_analysis.py',         '📋'),
    ('ML Pipeline',     'python modelo_xgboost.py',                   '🧠'),
    ('Feature Store',   'python feature_store.py',                     '🗄️'),
    ('Stress Test',     'python stress_test.py',                       '🏋️'),
    ('Técnicos',        'python analisis_tecnico.py',                  '📐'),
    ('Sentimiento',     'python analisis_social.py',                   '💬'),
    ('Macro',           'python macro_features.py',                    '🌍'),
    ('Streaming',       'python streaming_pipeline.py',                '🔴'),
]


class LauncherApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title('Agente Financiero v2.0')
        self.root.geometry('680x520')
        self.root.resizable(False, False)
        self.process = None
        self._build_ui()

    def _build_ui(self):
        # Header
        header = tk.Frame(self.root, bg='#0a0a2e')
        header.pack(fill='x')
        tk.Label(header, text='💼  AGENTE FINANCIERO', font=('Segoe UI', 16, 'bold'),
                 fg='#00d4ff', bg='#0a0a2e', pady=12).pack()
        tk.Label(header, text=f'Sistema Automatizado v2.0  |  {datetime.now().strftime("%Y-%m-%d %H:%M")}',
                 font=('Segoe UI', 9), fg='#888', bg='#0a0a2e').pack()

        # Buttons grid
        main = tk.Frame(self.root, bg='#111122')
        main.pack(fill='both', expand=True, padx=15, pady=10)

        canvas = tk.Canvas(main, bg='#111122', highlightthickness=0)
        scrollbar = ttk.Scrollbar(main, orient='vertical', command=canvas.yview)
        scroll_frame = tk.Frame(canvas, bg='#111122')

        scroll_frame.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox('all')))
        canvas.create_window((0, 0), window=scroll_frame, anchor='nw')
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')

        for i, (name, cmd, icon) in enumerate(MODULOS):
            row, col = divmod(i, 3)
            btn = tk.Button(scroll_frame, text=f'{icon}  {name}', font=('Segoe UI', 11),
                            bg='#1a1a3e', fg='#fff', activebackground='#2a2a5e',
                            activeforeground='#00d4ff', bd=0, padx=10, pady=8,
                            width=18, cursor='hand2',
                            command=lambda c=cmd, n=name: self._launch(c, n))
            btn.grid(row=row, column=col, padx=5, pady=5, sticky='ew')

        # Status bar
        self.status_var = tk.StringVar(value='✅  Listo')
        status = tk.Label(self.root, textvariable=self.status_var, font=('Segoe UI', 9),
                          fg='#aaa', bg='#0a0a2e', pady=4)
        status.pack(fill='x')

        # Footer
        footer = tk.Frame(self.root, bg='#0a0a2e')
        footer.pack(fill='x')
        tk.Button(footer, text='Abrir carpeta Datos', font=('Segoe UI', 9),
                  bg='#2a2a5e', fg='#fff', bd=0, padx=8, pady=3, cursor='hand2',
                  command=lambda: os.startfile(str(BASE / 'Datos'))).pack(side='left', padx=10, pady=5)
        tk.Label(footer, text='v2.0', font=('Segoe UI', 8), fg='#555', bg='#0a0a2e').pack(side='right', padx=10)
        tk.Button(footer, text='Salir', font=('Segoe UI', 9),
                  bg='#5a1a1a', fg='#fff', bd=0, padx=8, pady=3, cursor='hand2',
                  command=self._quit).pack(side='right', pady=5)

    def _launch(self, cmd, name):
        if self.process and self.process.poll() is None:
            if not messagebox.askyesno('Proceso activo', f'{name} ya está corriendo. ¿Detenerlo?'):
                return
            self._stop_process()

        self.status_var.set(f'▶️  Iniciando {name}...')
        self.root.update()

        def run():
            try:
                full_cmd = cmd
                if 'streamlit' in cmd:
                    # Extraer el archivo de destino (ej: dashboard_interactivo.py)
                    file_to_run = cmd.split()[-1]
                    # Construir ruta absoluta
                    abs_file_path = BASE / file_to_run
                    
                    port = os.environ.get('PORT', '8501')
                    full_cmd = f'python -m streamlit run "{abs_file_path}" --server.port {port} --server.address 0.0.0.0'
                    
                    threading.Thread(target=lambda: webbrowser.open(f'http://localhost:{port}'), daemon=True).start()
                
                self.process = subprocess.Popen(full_cmd, shell=True)
                self.process.wait()
                self.status_var.set('✅  Listo')
            except Exception as e:
                self.status_var.set(f'❌  Error: {e}')

        threading.Thread(target=run, daemon=True).start()

    def _stop_process(self):
        if self.process and self.process.poll() is None:
            self.process.terminate()
            self.process = None
            self.status_var.set('⏹️  Proceso detenido')

    def _quit(self):
        self._stop_process()
        self.root.destroy()

    def run(self):
        self.root.mainloop()


def main():
    if not HAS_TK:
        print(color('  [!] Tkinter no disponible. Usando modo terminal.\n', 'red'))
        from start import main as term_main
        term_main()
        return
    app = LauncherApp()
    app.run()


if __name__ == '__main__':
    main()