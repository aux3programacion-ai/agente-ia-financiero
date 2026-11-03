# VALIDACIÓN — qué hay y qué falta antes de dinero real

Estado a 2026-11-02 tras la puesta en orden del pipeline.

## Metodología (end-to-end, sin mirar el futuro)

- Target relativo al SPY a 20 días, features con embargo, walk-forward purgado.
- Walk-forward global (`ml.walk_forward`): **5 splits, test=63d, gap=20d** (>= horizonte),
  min_train=100. Implementado en `walkforward_validator.py`.
- Backtest: ejecución T+1 apertura, 10 bps de coste (5 comis + 5 slippage), benchmark buy&hold.

## Números honestos actuales

| Métrica | Valor | Nota |
|---|---|---|
| Walk-forward global (5 splits, gap 20d) | Acc 0.695 · AUC 0.733 · Brier 0.201 | Target: "la IA acertará el signo" (no es dirección pura). ~315 obs OOS |
| Aciertos direccionales 30d | **65.8%** (395/600) | Mejor señal del sistema, aún muestra limitada |
| Precisión global ponderada | 52.4% (9,716/19,492) | Consistente con ~azar; varía por sector/rango |
| Precisión AAPL único (2025, backtest.json) | −7.2%, Sharpe −0.62 | Negative alpha vs bench (+12.0%) |
| Backtest 30 tickers (2 años) | +27.5%, Sharpe 0.64 | Benchmark eq-weight: +102%, Sharpe ~1 |
| Convicción vs igual-peso B&H | B&H gana (Sharpe 4.03 vs 2.21) | Sin edge sobre el beta simple |
| Stress histórico | 2008: −26.9% / DD −35.8% · 2020: −14.3% / DD −34.1% · 2022: −32.0% / DD −31.8% | Cartera cae de verdad en crisis |
| Monte Carlo (10k sims) | VaR95≈$76k sobre $100k, P(pérdida)≈33% | Riesgo asimétrico |

## Conclusión

**No hay evidencia de edge comercial.** Hay piezas metodológicamente sanas (embargo, T+1,
costes, calibración isotónica), pero:
- el backtest real gana menos que comprar el índice / igual-peso,
- la calibración global ≈ moneda al aire,
- el "edge" aparente vive en muestras pequeñas (600 aciertos 30d, 315 OOS).

## Qué haría falta antes de capital real

1. Superar B&H equiponderado **neta de costes** en walk-forward multi-año por régimen (alcista/bajista/lateral).
2. Significancia estadística OOS (no solo point estimates): ≥1000 obs por split, IC bootstrap.
3. 6–12 meses de paper trading en verde vs B&H, con drawdown controlado < 15%.
4. Sobrepasar stress tests sin romper el límite de pérdida definido.
5. Calibración (Brier/ECE) estable por régimen y por sector, no solo global.

Hasta entonces: **solo análisis y paper trading, no dinero real.**
