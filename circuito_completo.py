#!/usr/bin/env python3
"""
circuito_completo.py - Orquestador del pipeline completo.
Integra todos los módulos: datos -> features -> ML -> calibración -> backtest -> modelo -> deploy.
"""
import json
import os
import time
import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional

DATA_DIR = os.environ.get('GITHUB_WORKSPACE', '.')
sys.path.insert(0, DATA_DIR)

OUTPUT_DIR = Path(DATA_DIR) / 'Datos'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TICKERS_CORE = ['NVDA','MU','DELL','AVGO','DDOG','SMCI','SNOW','CRWD','NOW','TSM',
                'ARM','OKTA','HPE','NTAP','CLS','AAPL','AMZN','GOOGL','META','MSFT',
                'LLY','AMAT','LRCX','PANW','ORCL','HON','UBER','GE','COST','NEE']

PIPELINE_STATE_PATH = OUTPUT_DIR / 'pipeline_state.json'


class PipelineOrchestrator:
    def __init__(self):
        self.start_time = time.time()
        self.steps = []
        self.state = self._load_state()

    def _load_state(self) -> Dict:
        if PIPELINE_STATE_PATH.exists():
            try:
                return json.loads(PIPELINE_STATE_PATH.read_text())
            except:
                pass
        return {'last_run': None, 'steps': {}, 'errors': []}

    def _save_state(self):
        self.state['last_run'] = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
        PIPELINE_STATE_PATH.write_text(json.dumps(self.state, indent=2))

    def _step(self, name: str, func, *args, **kwargs) -> Any:
        """Ejecuta un paso del pipeline con timing y error handling."""
        step_start = time.time()
        print(f'\n{"="*60}')
        print(f'  [{datetime.now().strftime("%H:%M:%S")}] {name}...')
        print(f'{"="*60}')
        
        try:
            result = func(*args, **kwargs)
            elapsed = time.time() - step_start
            status = 'OK' if result is not None and result != {} else 'WARN'
            self.state['steps'][name] = {
                'status': status,
                'elapsed_s': round(elapsed, 2),
                'timestamp': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
            }
            print(f'  [{status}] {name} completado en {elapsed:.1f}s')
            return result
        except Exception as e:
            elapsed = time.time() - step_start
            self.state['steps'][name] = {
                'status': 'ERROR',
                'elapsed_s': round(elapsed, 2),
                'error': str(e)[:200],
                'timestamp': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
            }
            self.state['errors'].append({
                'step': name,
                'error': str(e)[:500],
                'timestamp': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
            })
            print(f'  [ERROR] {name}: {str(e)[:150]}')
            return None

    def step_01_tecnicos(self):
        from analisis_tecnico import obtener_tecnicos
        # Monkey-patch yfinance cache
        try:
            from yfinance_cache import monkey_patch_yfinance
            monkey_patch_yfinance()
        except ImportError:
            pass
        return obtener_tecnicos()

    def step_02_social(self):
        from analisis_social import resultados
        return resultados

    def step_03_analyst_ratings(self):
        from analyst_ratings import main
        main.__globals__['TICKERS_CORE'] = TICKERS_CORE
        return main()

    def step_04_riesgo(self):
        import analisis_riesgo
        return analisis_riesgo.result

    def step_05_auto_features(self):
        from auto_feature_engineering import main
        return main()

    def step_06_auto_feature_discovery(self):
        from auto_feature_discovery import main
        return main()

    def step_07_aprendizaje(self):
        from aprendizaje import calibracion
        return calibracion

    def step_08_aprendizaje_skills(self):
        from aprendizaje_skills import main
        return main()

    def step_09_auto_evolution(self):
        from auto_evolution import main
        return main()

    def step_10_calibracion_real(self):
        from calibration import get_calibration_manager
        cm = get_calibration_manager()
        
        pred_path = OUTPUT_DIR / 'predicciones_hist.json'
        if pred_path.exists():
            predicciones = json.loads(pred_path.read_text())
            for ticker, data in predicciones.items():
                preds = data.get('predicciones', [])
                if len(preds) >= 10:
                    y_true = np.array([1 if p.get('acertada') else 0 for p in preds if p.get('acertada') is not None])
                    y_prob = np.array([p.get('probabilidad', 50) / 100.0 for p in preds if p.get('acertada') is not None])
                    if len(y_true) >= 50:
                        cm.calibrate(ticker, y_true, y_prob)
        
        print(f'[Calibracion] {len(cm.calibrators)} tickers calibrados')
        return cm.get_summary()

    def step_11_walkforward_validation(self):
        from walkforward_validator import cross_validate_model_predictive_power
        from xgboost import XGBClassifier
        
        # Cargar features y targets
        feat_path = OUTPUT_DIR / 'auto_features.json'
        pred_path = OUTPUT_DIR / 'predicciones_hist.json'
        regime_path = OUTPUT_DIR / 'analisis_tecnico.json'
        
        if not all(p.exists() for p in [feat_path, pred_path, regime_path]):
            print('  [!] Faltan archivos para walk-forward')
            return None
        
        try:
            features = json.loads(feat_path.read_text())
            predicciones = json.loads(pred_path.read_text())
            tecnicos = json.loads(regime_path.read_text())
        except:
            return None
        
        records = []
        for ticker, preds in predicciones.items():
            for p in preds.get('predicciones', []):
                fecha = p.get('fecha', '')[:10]
                if not fecha or ticker not in features.get('tickers', {}):
                    continue
                record = {'ticker': ticker, 'fecha': fecha, 'target': 1 if p.get('acertada') else 0}
                feat_dict = features['tickers'][ticker]
                for k, v in feat_dict.items():
                    if isinstance(v, (int, float)):
                        record[k] = v
                records.append(record)
        
        if len(records) < 50:
            print(f'  [!] Pocos registros: {len(records)}')
            return None
        
        import pandas as pd
        import numpy as np
        df = pd.DataFrame(records)
        feature_cols = [c for c in df.columns if c not in ('ticker', 'fecha', 'target')]
        
        X = df[feature_cols].values
        y = df['target'].values
        
        def xgb_fn(X, y, params):
            return XGBClassifier(**{**params}, n_estimators=50, max_depth=3, verbosity=0).fit(X, y)
        
        result = cross_validate_model_predictive_power(
            features_df=pd.DataFrame(X, columns=feature_cols),
            target_series=pd.Series(y),
            model_name='xgboost_global',
            model_fn=xgb_fn,
            n_splits=3,
            test_size_dias=42
        )
        
        return result

    def step_12_online_learning(self):
        from online_learning import get_online_learning_manager
        
        olm = get_online_learning_manager()
        pred_path = OUTPUT_DIR / 'predicciones_hist.json'
        
        if pred_path.exists():
            import numpy as np
            predicciones = json.loads(pred_path.read_text())
            for ticker, data in predicciones.items():
                for p in data.get('predicciones', []):
                    if p.get('acertada') is not None:
                        features = {k: v for k, v in p.items() if isinstance(v, (int, float)) and k not in ('probabilidad', 'confianza', 'acertada')}
                        target = 1 if p['acertada'] else 0
                        olm.update(ticker, features, target)
        
        summary = olm.summary()
        print(f'[Online] {summary["n_learners"]} learners, avg_acc={summary["avg_rolling_accuracy"]:.3f}')
        return summary

    def step_13_backtest(self):
        from backtest_engine import run_backtest
        end = datetime.now().strftime('%Y-%m-%d')
        start = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
        return run_backtest(start_date=start, end_date=end)

    def step_14_stress_test(self):
        from stress_test import StressTester
        st = StressTester()
        stress = st.run_historical_stress(TICKERS_CORE)
        mc = st.run_monte_carlo(expected_return=0.12, expected_vol=0.22)
        mc_stress = st.run_monte_carlo(expected_return=0.12, expected_vol=0.22, stress_factor=3.0)
        return {'stress': stress, 'monte_carlo': [mc, mc_stress]}

    def step_15_ablation(self):
        from ablation_studio import AblationStudio
        studio = AblationStudio()
        results = studio.run_trial(simulate_evaluator, n_trials=3)
        studio.generate_recommendations()
        studio.print_report()
        return studio.export_for_dashboard()

    def step_16_shap_explanations(self):
        from shap_explanations import explain_model_predictions
        
        model_path = OUTPUT_DIR / 'models' / 'xgboost_global' / 'global'
        if not model_path.exists():
            print('  [!] No hay modelo guardado para SHAP')
            return None
        
        try:
            import joblib
            import glob
            pkl_files = list(model_path.rglob('**/model.pkl'))
            if not pkl_files:
                return None
            model = joblib.load(pkl_files[-1])
            
            feat_path = OUTPUT_DIR / 'auto_features.json'
            if not feat_path.exists():
                return None
            
            features = json.loads(feat_path.read_text())
            features_list = []
            feature_names = []
            for ticker, feats in features.get('tickers', {}).items():
                if isinstance(feats, dict):
                    row = []
                    for k, v in feats.items():
                        if isinstance(v, (int, float)):
                            if k not in feature_names:
                                feature_names.append(k)
                            row.append(v)
                    if row:
                        features_list.append(row)
            
            if not features_list:
                return None
            
            import numpy as np
            X = np.array(features_list)
            if X.shape[1] != len(feature_names):
                feature_names = feature_names[:X.shape[1]]
            
            result = explain_model_predictions(model, feature_names, X)
            return result
        except Exception as e:
            print(f'  [!] SHAP: {e}')
            return None

    def step_17_regime_models(self):
        from regime_models import RegimeModelManager
        manager = RegimeModelManager()
        return manager.get_regime_performance()

    def run_all(self, steps: Optional[list] = None):
        """Ejecuta pipeline completo o steps específicos."""
        all_steps = [
            ('01. Técnicos', self.step_01_tecnicos),
            ('02. Social', self.step_02_social),
            ('03. Analyst Ratings', self.step_03_analyst_ratings),
            ('04. Riesgo', self.step_04_riesgo),
            ('05. Auto Features', self.step_05_auto_features),
            ('06. Feature Discovery', self.step_06_auto_feature_discovery),
            ('07. Aprendizaje', self.step_07_aprendizaje),
            ('08. Skills', self.step_08_aprendizaje_skills),
            ('09. Auto Evolution', self.step_09_auto_evolution),
            ('10. Calibración Real', self.step_10_calibracion_real),
            ('11. Walk-Forward', self.step_11_walkforward_validation),
            ('12. Online Learning', self.step_12_online_learning),
            ('13. Backtest', self.step_13_backtest),
            ('14. Stress Test', self.step_14_stress_test),
            ('15. Ablation Studio', self.step_15_ablation),
            ('16. SHAP Explanations', self.step_16_shap_explanations),
            ('17. Regime Models', self.step_17_regime_models),
        ]
        
        if steps:
            all_steps = [s for s in all_steps if s[0].split('.')[0].strip() in steps]
        
        results = {}
        for name, func in all_steps:
            results[name] = self._step(name, func)
        
        total_elapsed = time.time() - self.start_time
        errors = [s for s in self.state['steps'].values() if s.get('status') == 'ERROR']
        
        print(f'\n{"="*60}')
        print(f'  PIPELINE COMPLETO - RESUMEN')
        print(f'{"="*60}')
        print(f'  Total: {total_elapsed:.1f}s')
        print(f'  Steps OK: {sum(1 for s in self.state["steps"].values() if s["status"] == "OK")}')
        print(f'  Steps WARN: {sum(1 for s in self.state["steps"].values() if s["status"] == "WARN")}')
        print(f'  Steps ERROR: {len(errors)}')
        if errors:
            for e in errors:
                print(f'    ! {e}')
        print(f'{"="*60}')
        
        self._save_state()
        return results


def simulate_evaluator(components):
    """Evaluador simulado para ablation studio."""
    np.random.seed(42 + hash(frozenset(components)) % 1000)
    impact_map = {
        'ensemble_llm': 0.03, 'ensemble_specialization': 0.02,
        'debate_multimodelo': 0.01, 'skill_injection': 0.015,
        'mistake_injection': 0.01, 'few_shot_examples': 0.005,
        'meta_prompt_evolution': 0.01, 'auto_prompt_ab': 0.008,
        'calibracion_stonica': 0.02, 'xgb_blending': 0.025,
        'stacking_ensemble': 0.02, 'automl_winner': 0.015,
        'regime_models': 0.01, 'walk_forward': 0.005,
        'sentiment_vader': 0.005, 'online_learning': 0.015,
        'knowledge_distillation': 0.005
    }
    base = 0.55
    acc = base + sum(impact_map.get(c, 0) for c in components) + np.random.randn() * 0.005
    return {'accuracy': float(np.clip(acc, 0, 1))}


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Pipeline completo del Agente Financiero')
    parser.add_argument('--steps', nargs='*', help='Steps específicos (ej: 1 2 3)')
    parser.add_argument('--quick', action='store_true', help='Solo steps esenciales (1-10)')
    args = parser.parse_args()
    
    orchestrator = PipelineOrchestrator()
    
    if args.quick:
        results = orchestrator.run_all(steps=['1', '2', '3', '4', '5', '7', '8', '9', '10'])
    elif args.steps:
        results = orchestrator.run_all(steps=args.steps)
    else:
        results = orchestrator.run_all()
    
    return results


if __name__ == '__main__':
    main()