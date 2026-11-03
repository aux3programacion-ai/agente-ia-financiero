# Agente Financiero

Sistema de análisis y paper trading para un universo de 30 tickers (IA/semiconductores/software).
Pipeline: datos (yfinance, noticias) → features → ensemble XGBoost/LightGBM/CatBoost/RF →
calibración isotónica → backtest (T+1, 10 bps) → reportes/dashboard.

## Aviso de honestidad

No hay evidencia de edge predictivo: la precisión out-of-sample es ~52% y el backtest
mide rendimiento real vs benchmark (ver `Datos/backtest.json`). No operar dinero real con esta señal.

## Desactivar LLM (kill-switch)

El análisis LLM (OpenRouter/Gemini/NVIDIA) es opcional. Si está desactivado, el sistema cae
directamente al **trend-fallback** (probabilidades desde datos técnicos) escribiendo los mismos JSON.

Dos vías, ambas sin tocar código:

1. `config/settings.yaml`:
   ```yaml
   llm:
     enabled: false   # default: true
   ```
2. Variable de entorno: `LLM_DISABLED=1`

La llamada LLM central (`llamar_modelo` en `analisis_ia.py`) falla de forma controlada y
todos los sitios la capturan; el pipeline usa el fallback técnico.

## Salud del pipeline

`pipeline_state.json` (`Datos/`) registra el estado por step tras cada ejecución.
Los steps son: 01 técnicos, 05 auto features, 06 feature discovery, 07 aprendizaje,
10 calibración, 11 walk-forward, 13 backtest, 14 stress, 15/16 análisis, 17 régimen.

## Tests

```powershell
python -m pytest tests -q
```

## Flujo de trabajo

- Los datos (`Datos/`) y reportes (`Reportes/`) son generados; en GitHub los confirma el CI (`[auto] 30T ...`).
- Código comprometido pasa por el gate del workflow (compileall + ruff E9/F821/F811 + pytest).
