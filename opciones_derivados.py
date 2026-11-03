#!/usr/bin/env python3
"""opciones_derivados.py - Opciones, derivados, Greeks en tiempo real, estrategias multi-pata, y exportacion a dashboard."""
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

DATA_DIR = os.environ.get("GITHUB_WORKSPACE", ".")
OUTPUT_DIR = Path(DATA_DIR) / "Datos" / "opciones"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

try:
    from persistent_db import db
except Exception:
    db = None

try:
    import yfinance as yf
    YF_AVAILABLE = True
except ImportError:
    YF_AVAILABLE = False

@dataclass
class OptionGreeks:
    delta: float; gamma: float; theta: float; vega: float; rho: float
    def to_dict(self): return {k: float(v) for k,v in self.__dict__.items()}

@dataclass
class OptionPrice:
    premium: float; intrinsic: float; time_value: float
    greeks: OptionGreeks; implied_vol: float; model: str = "black_scholes"
    def to_dict(self):
        d = {k: float(v) for k,v in self.__dict__.items() if k != "greeks"}
        d["greeks"] = self.greeks.to_dict(); return d

class BlackScholes:
    @staticmethod
    def price(S, K, T, r, sigma, option_type="call"):
        d1 = (np.log(S/K)+(r+0.5*sigma**2)*T)/(sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        is_call = option_type == "call"
        if is_call:
            premium = S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
            delta = norm.cdf(d1); theta = (-S*norm.pdf(d1)*sigma/(2*np.sqrt(T))-r*K*np.exp(-r*T)*norm.cdf(d2))
            rho = K*T*np.exp(-r*T)*norm.cdf(d2)
        else:
            premium = K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
            delta = -norm.cdf(-d1); theta = (-S*norm.pdf(d1)*sigma/(2*np.sqrt(T))+r*K*np.exp(-r*T)*norm.cdf(-d2))
            rho = -K*T*np.exp(-r*T)*norm.cdf(-d2)
        gamma = norm.pdf(d1)/(S*sigma*np.sqrt(T))
        vega = S*norm.pdf(d1)*np.sqrt(T)
        intrinsic = max(S-K,0) if is_call else max(K-S,0)
        return OptionPrice(float(premium),float(intrinsic),float(premium-intrinsic),
            OptionGreeks(float(delta),float(gamma),float(theta),float(vega),float(rho)),sigma)
    @staticmethod
    def implied_volatility(mp, S, K, T, r, option_type="call", tol=1e-6, max_iter=100):
        sigma = 0.3
        for _ in range(max_iter):
            opt = BlackScholes.price(S, K, T, r, sigma, option_type)
            diff = opt.premium - mp
            if abs(diff) < tol: return sigma
            if opt.greeks.vega < 1e-10: break
            sigma = sigma - diff/(opt.greeks.vega*100)
            sigma = max(0.01, min(2.0, sigma))
        return sigma

class VolatilitySmile:
    def __init__(self): self.smile_data = {}
    def compute_smile(self, strikes, prices, S, T, r, option_type="call"):
        results = []
        for K,mp in zip(strikes,prices):
            iv = BlackScholes.implied_volatility(mp,S,K,T,r,option_type)
            results.append({"strike":float(K),"imp_vol":float(iv),"moneyness":float(np.log(K/S))})
        self.smile_data = pd.DataFrame(results).to_dict("records")
        return pd.DataFrame(results)

class DeltaHedger:
    def __init__(self, S, K, T, r, sigma, option_type="call"):
        self.S=S;self.K=K;self.T=T;self.r=r;self.sigma=sigma
        self.option_type=option_type;self.hedge=0;self.cash=0;self.trades=[]
    def rebalance(self, new_S, dt):
        rem_T = max(self.T-dt, 0.001)
        opt = BlackScholes.price(new_S,self.K,rem_T,self.r,self.sigma,self.option_type)
        target_delta = opt.greeks.delta
        hedge_needed = -target_delta*100
        d_hedge = hedge_needed - self.hedge
        cost = d_hedge*new_S; self.cash -= cost
        self.hedge = hedge_needed; self.S = new_S; self.T = rem_T
        self.trades.append({"S":float(new_S),"delta":float(target_delta),"change":int(d_hedge),
            "cost":float(cost),"cash":float(self.cash)})
        opt2 = BlackScholes.price(new_S,self.K,rem_T,self.r,self.sigma,self.option_type)
        pnl = opt2.premium*100 + self.hedge*new_S + self.cash
        return {"delta":float(target_delta),"hedge":int(self.hedge),"cash":float(self.cash),"pnl":float(pnl)}

class MultiLegStrategy:
    @staticmethod
    def bull_call_spread(S, K1, K2, T, r, sigma):
        c1=BlackScholes.price(S,K1,T,r,sigma,"call");c2=BlackScholes.price(S,K2,T,r,sigma,"call")
        return [{"leg":"long_call","strike":K1,"premium":c1.premium},{"leg":"short_call","strike":K2,"premium":c2.premium}]
    @staticmethod
    def straddle(S, K, T, r, sigma):
        c=BlackScholes.price(S,K,T,r,sigma,"call");p=BlackScholes.price(S,K,T,r,sigma,"put")
        return [{"leg":"long_call","strike":K,"premium":c.premium},{"leg":"long_put","strike":K,"premium":p.premium}]
    @staticmethod
    def iron_condor(S, K1, K2, K3, K4, T, r, sigma):
        return [{"leg":"long_put","strike":K1},{"leg":"short_put","strike":K2},
                {"leg":"short_call","strike":K3},{"leg":"long_call","strike":K4}]

class OptionsAnalyzer:
    def __init__(self): self.results = {}
    def analyze_chain(self, S, strikes, T, r, sigma):
        rows=[]
        for K in strikes:
            c=BlackScholes.price(S,K,T,r,sigma,"call");p=BlackScholes.price(S,K,T,r,sigma,"put")
            rows.append({"strike":float(K),"call_premium":float(c.premium),"put_premium":float(p.premium),
                "call_delta":float(c.greeks.delta),"put_delta":float(p.greeks.delta)})
        df=pd.DataFrame(rows);self.results["chain"]=df.to_dict("records");return df
    def save_report(self):
        (OUTPUT_DIR/"options_analysis.json").write_text(json.dumps(self.results,indent=2),encoding="utf-8")

class OptionsRealDataLoader:
    def fetch_spot_price(self, ticker: str) -> float | None:
        if YF_AVAILABLE:
            try:
                t = yf.Ticker(ticker)
                return float(t.history(period="1d")["Close"].iloc[-1])
            except Exception:
                pass
        return None

    def fetch_option_chain(self, ticker: str, expiration: str | None = None):
        if not YF_AVAILABLE:
            return None
        try:
            t = yf.Ticker(ticker)
            exp = expiration or t.options[0] if t.options else None
            if not exp:
                return None
            chain = t.option_chain(exp)
            calls = chain.calls.copy()
            puts = chain.puts.copy()
            calls["type"] = "call"
            puts["type"] = "put"
            df = pd.concat([calls, puts], ignore_index=True)
            df["ticker"] = ticker
            return df
        except Exception:
            return None

    def analyze_real_ticker(self, ticker: str, r: float = 0.05):
        spot = self.fetch_spot_price(ticker)
        if spot is None:
            print(f"  [!] No se pudo obtener precio para {ticker}, usando datos simulados")
            spot = 100.0
        chain = self.fetch_option_chain(ticker)
        results = {"ticker": ticker, "spot": spot, "timestamp": datetime.now().isoformat()}
        strikes = np.arange(spot * 0.8, spot * 1.2, spot * 0.05)
        sigma = 0.25
        T = 30 / 365
        rows = []
        for K in strikes:
            c = BlackScholes.price(spot, K, T, r, sigma, "call")
            p = BlackScholes.price(spot, K, T, r, sigma, "put")
            rows.append({"strike": round(K, 2), "call_premium": round(c.premium, 2),
                         "put_premium": round(p.premium, 2),
                         "call_delta": round(c.greeks.delta, 4), "put_delta": round(p.greeks.delta, 4),
                         "call_iv": round(c.implied_vol, 4), "put_iv": round(p.implied_vol, 4)})
        results["chain"] = rows
        smile = VolatilitySmile()
        smile.compute_smile(strikes, [r["call_premium"] for r in rows], spot, T, r)
        results["smile"] = smile.smile_data
        return results

    def save_analysis(self, results: dict):
        path = OUTPUT_DIR / f"options_{results['ticker']}_{datetime.now().strftime('%Y%m%d')}.json"
        path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
        if db:
            try:
                db.guardar_metrica("options", "spot", results["spot"], {"ticker": results["ticker"]})
                db.guardar_metrica("options", "n_strikes", len(results.get("chain", [])), {"ticker": results["ticker"]})
            except Exception:
                pass
        return str(path)


class StrategyBacktester:
    def __init__(self, capital: float = 100000):
        self.capital = capital
        self.results = {}

    def backtest_straddle(self, S, K, T, r, sigma, n_days: int = 30, moves: np.ndarray | None = None):
        if moves is None:
            moves = np.random.randn(n_days) * sigma * S / np.sqrt(252)
        path = [S]
        for m in moves:
            path.append(path[-1] + m)
        daily_pnl = []
        for i, price in enumerate(path[:-1]):
            rem_T = max(T - (i + 1) / 252, 0.001)
            c = BlackScholes.price(price, K, rem_T, r, sigma, "call")
            p = BlackScholes.price(price, K, rem_T, r, sigma, "put")
            pnl = (c.premium + p.premium) * 100
            daily_pnl.append(pnl)
        result = {"initial_cost": daily_pnl[0] if daily_pnl else 0,
                  "final_value": daily_pnl[-1] if daily_pnl else 0,
                  "total_pnl": (daily_pnl[-1] - daily_pnl[0]) if len(daily_pnl) > 1 else 0,
                  "max_pnl": max(daily_pnl) if daily_pnl else 0,
                  "min_pnl": min(daily_pnl) if daily_pnl else 0}
        path_str = OUTPUT_DIR / f"straddle_backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        path_str.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
        return result


def analyze_ticker(ticker: str, save: bool = True):
    loader = OptionsRealDataLoader()
    result = loader.analyze_real_ticker(ticker)
    if save:
        path = loader.save_analysis(result)
        print(f"  Analisis guardado en: {path}")
    return result


if __name__ == "__main__":
    S,K,T,r,sigma=100,100,1.0,0.05,0.25
    opt=BlackScholes.price(S,K,T,r,sigma,"call")
    print(f"Call: {opt.premium:.4f}, Delta: {opt.greeks.delta:.4f}")
    strikes=np.arange(80,121,5)
    smile=VolatilitySmile()
    df=smile.compute_smile(strikes,np.ones(len(strikes))*10,S,T,r)
    print(f"Smile: {len(df)} strikes")
    condor=MultiLegStrategy.iron_condor(S,85,95,105,115,T,r,sigma)
    print(f"Iron Condor: {len(condor)} legs")
