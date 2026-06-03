import json
import os
import datetime
import urllib.request
import re
import time
import calendar


def es_dia_habil(fecha):
    return fecha.weekday() < 5


def prox_dia_habil(fecha):
    while not es_dia_habil(fecha):
        fecha += datetime.timedelta(days=1)
    return fecha


def obtener_fomc(ano, hoy, limite):
    reuniones = [
        (1, 28, 29), (3, 17, 18), (5, 6, 7),
        (6, 16, 17), (7, 28, 29), (9, 15, 16),
        (11, 3, 4), (12, 15, 16)
    ]
    eventos = []
    for mes, d1, d2 in reuniones:
        f = datetime.date(ano, mes, d2)
        if hoy <= f <= limite:
            eventos.append({
                "fecha": f.isoformat(),
                "tipo": "FOMC",
                "descripcion": "Decision de tasa de interes de la Reserva Federal (EEUU)",
                "impacto": "alto"
            })
    return eventos


def obtener_cpi(ano, hoy, limite):
    eventos = []
    for mes in range(1, 13):
        f = datetime.date(ano, mes, 14)
        f = prox_dia_habil(f)
        if hoy <= f <= limite:
            eventos.append({
                "fecha": f.isoformat(),
                "tipo": "CPI",
                "descripcion": "Indice de Precios al Consumidor (EEUU)",
                "impacto": "alto"
            })
    return eventos


def obtener_nfp(ano, hoy, limite):
    eventos = []
    for mes in range(1, 13):
        cal = calendar.monthcalendar(ano, mes)
        primer_viernes = None
        for semana in cal:
            if semana[calendar.FRIDAY] != 0:
                primer_viernes = semana[calendar.FRIDAY]
                break
        if primer_viernes is None:
            continue
        f = datetime.date(ano, mes, primer_viernes)
        if hoy <= f <= limite:
            eventos.append({
                "fecha": f.isoformat(),
                "tipo": "NFP",
                "descripcion": "Nomina No Agricola (EEUU)",
                "impacto": "alto"
            })
    return eventos


def obtener_gdp(ano, hoy, limite):
    meses = [1, 4, 7, 10]
    eventos = []
    for mes in meses:
        dias_en_mes = calendar.monthrange(ano, mes)[1]
        for d in range(dias_en_mes, 0, -1):
            f = datetime.date(ano, mes, d)
            if es_dia_habil(f):
                break
        if hoy <= f <= limite:
            eventos.append({
                "fecha": f.isoformat(),
                "tipo": "GDP",
                "descripcion": "Producto Interno Bruto (EEUU) - Avance Trimestral",
                "impacto": "alto"
            })
    return eventos


def obtener_earnings(ano, hoy, limite):
    ventanas = [
        (1, 10, 2, 7),
        (4, 10, 5, 7),
        (7, 10, 8, 7),
        (10, 10, 11, 7)
    ]
    eventos = []
    for mes_inicio, dia_inicio, mes_fin, dia_fin in ventanas:
        f = datetime.date(ano, mes_fin, dia_fin)
        if hoy <= f <= limite:
            eventos.append({
                "fecha": f.isoformat(),
                "tipo": "EARNINGS",
                "descripcion": "Temporada de Resultados Corporativos (EEUU)",
                "impacto": "alto"
            })
    return eventos


def main():
    try:
        hoy = datetime.datetime.now().date()
        limite = hoy + datetime.timedelta(days=60)
        ano = hoy.year

        todos = []
        todos.extend(obtener_fomc(ano, hoy, limite))
        todos.extend(obtener_cpi(ano, hoy, limite))
        todos.extend(obtener_nfp(ano, hoy, limite))
        todos.extend(obtener_gdp(ano, hoy, limite))
        todos.extend(obtener_earnings(ano, hoy, limite))

        if hoy.year != limite.year:
            todos.extend(obtener_fomc(limite.year, hoy, limite))
            todos.extend(obtener_cpi(limite.year, hoy, limite))
            todos.extend(obtener_nfp(limite.year, hoy, limite))
            todos.extend(obtener_gdp(limite.year, hoy, limite))
            todos.extend(obtener_earnings(limite.year, hoy, limite))

        todos.sort(key=lambda e: e["fecha"])

        tipos = {}
        for e in todos:
            t = e["tipo"]
            tipos[t] = tipos.get(t, 0) + 1

        partes = []
        for t, c in sorted(tipos.items()):
            partes.append(f"{c} {t}")
        resumen = "Proximos eventos economicos clave en los proximos 60 dias: " + ", ".join(partes)

        salida = {
            "timestamp": datetime.datetime.now().isoformat(),
            "proximos_eventos": todos,
            "resumen": resumen
        }

        ruta = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Datos", "calendario_economico.json")
        with open(ruta, "w", encoding="utf-8") as f:
            json.dump(salida, f, ensure_ascii=False, indent=2)

        print(f"Calendario economico generado: {len(todos)} eventos")
        for e in todos:
            print(f"  {e['fecha']} - {e['tipo']}: {e['descripcion']}")

    except Exception:
        ruta = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Datos", "calendario_economico.json")
        fallback = {
            "timestamp": datetime.datetime.now().isoformat(),
            "proximos_eventos": [],
            "resumen": "No se pudieron generar los eventos economicos"
        }
        with open(ruta, "w", encoding="utf-8") as f:
            json.dump(fallback, f, ensure_ascii=False, indent=2)
        print("Error generando calendario economico, se uso fallback")


if __name__ == "__main__":
    main()
