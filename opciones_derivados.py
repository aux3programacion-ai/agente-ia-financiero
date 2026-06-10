#!/usr/bin/env python3
"""opciones_derivados.py - Opciones y derivados financieros."""
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from dataclasses import dataclass
from scipy.stats import norm
import json, os
from pathlib import Path
from config.settings import get_setting
DATA_DIR = os.environ.get("GITHUB_WORKSPACE", ".")
OUTPUT_DIR = Path(DATA_DIR) / "Datos" / "opciones"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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

if __name__=="__main__":
    S,K,T,r,sigma=100,100,1.0,0.05,0.25
    opt=BlackScholes.price(S,K,T,r,sigma,"call")
    print(f"Call: {opt.premium:.4f}, Delta: {opt.greeks.delta:.4f}")
    strikes=np.arange(80,121,5)
    smile=VolatilitySmile()
    df=smile.compute_smile(strikes,np.ones(len(strikes))*10,S,T,r)
    print(f"Smile: {len(df)} strikes")
    condor=MultiLegStrategy.iron_condor(S,85,95,105,115,T,r,sigma)
    print(f"Iron Condor: {len(condor)} legs")
