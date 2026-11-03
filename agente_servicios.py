#!/usr/bin/env python3
"""
agente_servicios.py - Monitor de salud + servicios programados del Agente Financiero.
Ejecuta en puerto 8001 (uvicorn agente_servicios:app).
Endpoints:
  /api/v1/salud          - estado de todos los modulos y datos
  /api/v1/ejecutar/{mod} - ejecutar un modulo en subproceso
  /api/v1/tareas         - estado de las tareas programadas
  /api/v1/paper          - estado del paper trading
"""
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from logger import get_module_logger

log = get_module_logger('agente_servicios')

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

try:
    from apscheduler.schedulers.background import BackgroundScheduler
    SCHEDULER_AVAILABLE = True
except ImportError:
    SCHEDULER_AVAILABLE = False

AGENTE_DIR = Path(__file__).parent
DATA_DIR = AGENTE_DIR / 'Datos'
PYTHON = sys.executable

MODULOS = [
    "analisis_tecnico", "modelo_xgboost", "analisis_ia",
    "paper_trading", "trading_orchestrator", "optimizacion_portafolio",
    "analisis_riesgo", "analisis_patrones", "opciones",
    "regimen_mercado", "noticias_mercado", "multi_agent_system",
]

ARCHIVOS_DATOS = [
    "analisis_ia.json", "analisis_riesgo.json", "optimizacion_portafolio.json",
    "modelo_xgboost.json", "regimen_mercado.json", "paper_trading/paper_trading_report.json",
]

app = None
if FASTAPI_AVAILABLE:
    app = FastAPI(title='Agente Financiero Servicios', version='1.0.0')

# ---------------- Salud ----------------

def check_module(modulo: str, timeout: int = 25) -> dict:
    """Verifica que un modulo importe correctamente en un subproceso (no bloquea)."""
    codigo = (
        "import importlib,sys;"
        f"sys.path.insert(0,{str(AGENTE_DIR)!r});"
        f"m=importlib.import_module({modulo!r});"
        "print('OK')"
    )
    try:
        resultado = subprocess.run(
            [PYTHON, "-c", codigo],
            capture_output=True, timeout=timeout, cwd=str(AGENTE_DIR)
        )
        if resultado.returncode == 0 and b"OK" in resultado.stdout:
            return {"estado": "ok", "detalle": "modulo importa correctamente"}
        detalle = (resultado.stderr or resultado.stdout).decode(errors="replace").strip().splitlines()
        return {"estado": "error", "detalle": detalle[-1][:120] if detalle else "fallo desconocido"}
    except subprocess.TimeoutExpired:
        return {"estado": "lento", "detalle": f"importacion >{timeout}s (modulo pesado)"}
    except Exception as e:
        return {"estado": "error", "detalle": str(e)[:120]}


def check_data_file(nombre: str, max_age_hours: float = 48) -> dict:
    p = DATA_DIR / nombre
    if not p.exists():
        return {"estado": "ausente", "detalle": "no existe"}
    edad_h = (time.time() - p.stat().st_mtime) / 3600
    if edad_h > max_age_hours:
        return {"estado": "antiguo", "detalle": f"actualizado hace {edad_h:.1f}h"}
    return {"estado": "ok", "detalle": f"actualizado hace {edad_h:.1f}h"}


def check_api_keys() -> dict:
    env_path = AGENTE_DIR / ".env"
    if not env_path.exists():
        return {"estado": "ausente", "detalle": "no existe .env"}
    keys = []
    try:
        for linea in env_path.read_text(encoding="utf-8", errors="replace").splitlines():
            linea = linea.strip()
            if linea and not linea.startswith("#") and "=" in linea:
                k = linea.split("=")[0].strip()
                v = linea.split("=", 1)[1].strip().strip('"').strip("'")
                estado = "configurada" if v else "VACIA"
                keys.append({"clave": k, "estado": estado})
    except Exception as e:
        return {"estado": "error", "detalle": str(e)[:120]}
    return {"estado": "ok", "detalle": keys}


@app.get("/api/v1/salud")
def salud():
    modulos = {m: check_module(m) for m in MODULOS}
    archivos = {n: check_data_file(n) for n in ARCHIVOS_DATOS}
    keys = check_api_keys()

    resumen = {
        "modulos_ok": sum(1 for v in modulos.values() if v["estado"] == "ok"),
        "modulos_total": len(modulos),
        "archivos_ok": sum(1 for v in archivos.values() if v["estado"] == "ok"),
        "archivos_total": len(archivos),
    }
    return {
        "servicio": "Agente Financiero Servicios",
        "timestamp": datetime.now().isoformat(),
        "resumen": resumen,
        "modulos": modulos,
        "datos": archivos,
        "api_keys": keys,
    }


# ---------------- Ejecutar modulos ----------------

class EjecucionRequest(BaseModel):
    modulo: str
    timeout: int = 120


@app.post("/api/v1/ejecutar")
def ejecutar(req: EjecucionRequest):
    if req.modulo not in MODULOS:
        raise HTTPException(404, f"Modulo desconocido: {req.modulo}")
    log.info(f"Ejecutando modulo: {req.modulo}")
    codigo = (
        "import sys,time;"
        f"sys.path.insert(0,{str(AGENTE_DIR)!r});"
        f"m=__import__({req.modulo!r});"
        "print('MODULO_EJECUTADO_OK')"
    )
    try:
        t0 = time.time()
        resultado = subprocess.run(
            [PYTHON, "-c", codigo], capture_output=True, timeout=req.timeout,
            cwd=str(AGENTE_DIR)
        )
        duracion = time.time() - t0
        if resultado.returncode == 0:
            return {"modulo": req.modulo, "estado": "ok", "duracion_s": round(duracion, 1)}
        detalle = (resultado.stderr or resultado.stdout).decode(errors="replace").strip().splitlines()
        return {"modulo": req.modulo, "estado": "error", "duracion_s": round(duracion, 1),
                "detalle": detalle[-1][:200] if detalle else "desconocido"}
    except subprocess.TimeoutExpired:
        return {"modulo": req.modulo, "estado": "timeout", "duracion_s": req.timeout}


# ---------------- Paper trading ----------------

_paper_state = {"activo": False, "ultima_actualizacion": None, "reporte": None}


def _actualizar_paper():
    """Ciclo de paper trading programado."""
    try:
        from paper_trading import get_paper_trading_manager
        m = get_paper_trading_manager()
        if not m.active:
            m.start()
            m.broker.set_price_provider(lambda t: 150.0)
        m.stop_loss_pct = 5.0
        # Generar senales simples desde el analisis tecnico guardado
        senales = {}
        ia_data = DATA_DIR / "analisis_ia.json"
        if ia_data.exists():
            try:
                data = json.loads(ia_data.read_text(encoding="utf-8"))
                for t, info in (data.get("probabilidades") or {}).items():
                    if isinstance(info, dict):
                        prob = info.get("probabilidad", 50)
                        senales[t] = {
                            "direction": "buy" if prob > 60 else ("sell" if prob < 40 else "hold"),
                            "confidence": info.get("confianza", 50),
                            "price": 150.0,
                        }
            except Exception:
                pass
        if senales:
            m.run_scheduled(lambda: senales, 60)
        reporte = m.generate_report() if hasattr(m, "generate_report") else None
        _paper_state["activo"] = m.active
        _paper_state["stop_loss_pct"] = m.stop_loss_pct
        _paper_state["stop_loss_triggered"] = getattr(m, "stop_loss_triggered", [])[-10:]
        _paper_state["ultima_actualizacion"] = datetime.now().isoformat()
        _paper_state["reporte"] = reporte
        log.info(f"Paper trading actualizado: {len(senales)} senales")
    except Exception as e:
        log.error(f"Error paper trading: {e}")


@app.get("/api/v1/paper")
def paper_status():
    _paper_state["configurado"] = True
    return _paper_state


# ---------------- Tareas programadas ----------------

scheduler = None
if SCHEDULER_AVAILABLE:
    scheduler = BackgroundScheduler()
    scheduler.add_job(_actualizar_paper, 'interval', hours=1, id='paper_horario', replace_existing=True)
    scheduler.start()

TAREAS = [
    {"id": "paper_horario", "nombre": "Paper trading horario", "intervalo": "1h"},
]


@app.get("/api/v1/tareas")
def tareas():
    estado = []
    for t in TAREAS:
        estado.append({
            "id": t["id"], "nombre": t["nombre"], "intervalo": t["intervalo"],
            "scheduler_activo": bool(scheduler and scheduler.running),
        })
    return {"tareas": estado, "scheduler_disponible": SCHEDULER_AVAILABLE}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
