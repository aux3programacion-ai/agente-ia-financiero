#!/usr/bin/env python3
"""
ablation_studio.py - Ablation study automatizado.
Prueba qué componentes del sistema aportan más valor.
Sugiere qué optimizaciones mantener, revisar, o eliminar.
"""
import json
import os
import time
import itertools
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Callable

from config.settings import get_setting

DATA_DIR = os.environ.get('GITHUB_WORKSPACE', '.')
OUTPUT_DIR = Path(DATA_DIR) / 'Datos'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

AB_CONFIG = get_setting('ablation', {})
N_TRIALS = AB_CONFIG.get('n_trials', 5)
OPTIMIZACIONES = AB_CONFIG.get('optimizaciones', [])
UMBRALES = AB_CONFIG.get('umbrales_recomendacion', {})


class AblationStudio:
    def __init__(self):
        self.results_path = OUTPUT_DIR / 'ablation_results.json'
        self.optimizaciones = OPTIMIZACIONES
        self._load()

    def _load(self):
        if self.results_path.exists():
            try:
                self.results = json.loads(self.results_path.read_text())
            except:
                self.results = {'trials': [], 'recommendations': [], 'timestamp': None}
        else:
            self.results = {'trials': [], 'recommendations': [], 'timestamp': None}

    def _save(self):
        self.results_path.write_text(json.dumps(self.results, indent=2))

    def run_trial(
        self,
        evaluator_fn: Callable[[List[str]], Dict[str, float]],
        components: Optional[List[str]] = None,
        baseline_name: str = 'full_system',
        n_trials: Optional[int] = None
    ) -> List[Dict]:
        """
        Ejecuta ablation trial: prueba el sistema con/sin cada componente.
        
        Args:
            evaluator_fn: Función que recibe lista de componentes activos y retorna
                         dict de métricas {metric_name: value}
            components: Lista de componentes a probar (default: self.optimizaciones)
            n_trials: Número de trials (default: config)
            
        Returns:
            Lista de resultados por trial
        """
        if components is None:
            components = self.optimizaciones
        if n_trials is None:
            n_trials = N_TRIALS
        
        results = []
        
        for trial in range(n_trials):
            print(f'[Ablation] Trial {trial + 1}/{N_TRIALS}...')
            
            fixed_seed = 42 + trial
            np.random.seed(fixed_seed)
            
            baseline_metrics = evaluator_fn(components)
            
            for comp in components:
                remaining = [c for c in components if c != comp]
                metrics_without = evaluator_fn(remaining)
                
                diff = {}
                for k in baseline_metrics:
                    if k in metrics_without:
                        diff[k] = metrics_without[k] - baseline_metrics[k]
                
                results.append({
                    'trial': trial,
                    'component': comp,
                    'baseline': baseline_metrics,
                    'without_component': metrics_without,
                    'diff': diff,
                    'avg_diff': float(np.mean(list(diff.values()))) if diff else 0,
                    'seed': fixed_seed
                })
        
        self.results['trials'] = results
        self._save()
        return results

    def _compute_component_impact(self, component: str, metric: str = 'accuracy') -> Dict:
        """Calcula el impacto promedio de un componente en una métrica."""
        diffs = [
            r['diff'].get(metric, 0)
            for r in self.results['trials']
            if r['component'] == component
        ]
        
        if not diffs:
            return {'mean': 0, 'std': 0, 'n': 0}
        
        return {
            'mean': float(np.mean(diffs)),
            'std': float(np.std(diffs)),
            'min': float(min(diffs)),
            'max': float(max(diffs)),
        'n': len(diffs),
        'consistent': bool(abs(np.mean(diffs)) > np.std(diffs))
        }

    def generate_recommendations(self, metric: str = 'accuracy') -> List[Dict]:
        """
        Genera recomendaciones basadas en los resultados de ablation.
        
        Returns:
            Lista de {component, impact, recommendation, detalles}
        """
        if not self.results['trials']:
            return []
        
        recommendations = []
        components_tested = set(r['component'] for r in self.results['trials'])
        
        for comp in sorted(components_tested):
            impact = self._compute_component_impact(comp, metric)
            
            if impact['mean'] > UMBRALES.get('critical', 0.02):
                rec = 'mantener'
                reason = 'CRITICAL - impacto positivo significativo'
            elif impact['mean'] > UMBRALES.get('important', 0.01):
                rec = 'mantener'
                reason = 'IMPORTANT - contribuye positivamente'
            elif impact['mean'] > UMBRALES.get('nice_to_have', 0.005):
                rec = 'nice_to_have'
                reason = 'Útil pero no crítico'
            elif impact['mean'] < UMBRALES.get('remove', -0.01):
                rec = 'remover'
                reason = 'NEGATIVO - empeora el sistema'
            elif impact['mean'] < UMBRALES.get('review', -0.005):
                rec = 'revisar'
                reason = 'DUDOSO - impacto negativo leve'
            else:
                rec = 'neutral'
                reason = 'NEUTRAL - sin impacto significativo'
            
            recommendations.append({
                'component': comp,
                'impact_mean': round(impact['mean'], 4),
                'impact_std': round(impact['std'], 4),
                'impact_consistent': impact['consistent'],
                'n_trials': impact['n'],
                'recommendation': rec,
                'reason': reason,
                'details': {
                    'min_impact': round(impact['min'], 4),
                    'max_impact': round(impact['max'], 4)
                }
            })
        
        recommendations.sort(key=lambda x: x['impact_mean'], reverse=True)
        
        self.results['recommendations'] = recommendations
        self.results['metric'] = metric
        self.results['updated'] = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
        self._save()
        
        return recommendations

    def print_report(self, metric: str = 'accuracy'):
        """Imprime reporte de ablation en consola."""
        if not self.results['recommendations']:
            self.generate_recommendations(metric)
        
        print(f'\n{"="*60}')
        print(f'  ABLATION STUDIO REPORT (metric: {metric.upper()})')
        print(f'{"="*60}')
        print(f'  Trials: {len(set(r["trial"] for r in self.results["trials"]))}')
        print(f'  Components tested: {len(set(r["component"] for r in self.results["trials"]))}')
        print(f'{"="*60}\n')
        
        emojis = {
            'mantener': '[KEEP]',
            'important': '[IMP]',
            'nice_to_have': '[NICE]',
            'revisar': '[REV]',
            'remover': '[REM]',
            'neutral': '[NEU]'
        }
        
        for rec in self.results['recommendations']:
            emoji = emojis.get(rec['recommendation'], '[?]')
            sign = '+' if rec['impact_mean'] > 0 else ''
            print(f'  {emoji} {rec["component"]:35s} {sign}{rec["impact_mean"]:.4f} ± {rec["impact_std"]:.4f}  | {rec["reason"]}')
        
        print(f'\n{"="*60}')
        
        # Summary stats
        criticals = [r for r in self.results['recommendations'] if r['recommendation'] == 'mantener']
        removes = [r for r in self.results['recommendations'] if r['recommendation'] == 'remover']
        reviews = [r for r in self.results['recommendations'] if r['recommendation'] == 'revisar']
        
        if criticals:
            print(f'\n  Mantener (críticos): {", ".join(r["component"] for r in criticals[:5])}')
        if removes:
            print(f'\n  Remover: {", ".join(r["component"] for r in removes[:5])}')
        if reviews:
            print(f'\n  Revisar: {", ".join(r["component"] for r in reviews[:5])}')
        
        print()

    def export_for_dashboard(self) -> Dict:
        """Exporta resultados para dashboard."""
        return {
            'n_trials': len(set(r['trial'] for r in self.results['trials'])),
            'n_components': len(set(r['component'] for r in self.results['trials'])),
            'recommendations': {
                'mantener': [r for r in self.results['recommendations'] if r['recommendation'] == 'mantener'],
                'remover': [r for r in self.results['recommendations'] if r['recommendation'] == 'remover'],
                'revisar': [r for r in self.results['recommendations'] if r['recommendation'] == 'revisar'],
                'neutral': [r for r in self.results['recommendations'] if r['recommendation'] == 'neutral']
            },
            'updated': self.results.get('updated')
        }


def simulate_evaluator(components: List[str]) -> Dict[str, float]:
    """
    Evaluador simulado para demostración.
    En producción, reemplazar con evaluación real del sistema.
    """
    np.random.seed(42 + hash(frozenset(components)) % 1000)
    
    base_acc = 0.55
    base_sharpe = 0.8
    
    # Cada componente tiene un impacto simulado
    impact_map = {
        'ensemble_llm': (0.03, 0.05),
        'ensemble_specialization': (0.02, 0.03),
        'debate_multimodelo': (0.01, 0.04),
        'skill_injection': (0.015, 0.02),
        'mistake_injection': (0.01, 0.01),
        'few_shot_examples': (0.005, 0.01),
        'meta_prompt_evolution': (0.01, 0.015),
        'auto_prompt_ab': (0.008, 0.01),
        'calibracion_stonica': (0.02, 0.03),
        'xgb_blending': (0.025, 0.04),
        'stacking_ensemble': (0.02, 0.035),
        'automl_winner': (0.015, 0.02),
        'regime_models': (0.01, 0.025),
        'walk_forward': (0.005, 0.01),
        'sentiment_vader': (0.005, 0.008),
        'online_learning': (0.015, 0.02),
        'knowledge_distillation': (0.005, 0.01),
        'auto_features': (0.01, 0.015),
        'time_series_features': (0.008, 0.012)
    }
    
    acc = base_acc
    sharpe = base_sharpe
    
    for comp in components:
        impacts = impact_map.get(comp, (0, 0))
        acc += impacts[0] + np.random.randn() * 0.005
        sharpe += impacts[1] + np.random.randn() * 0.01
    
    return {
        'accuracy': float(np.clip(acc, 0, 1)),
        'sharpe_ratio': float(max(sharpe, 0)),
        'win_rate': float(np.clip(acc + np.random.randn() * 0.02, 0, 1))
    }


if __name__ == '__main__':
    print('[Ablation] Running simulated ablation study...')
    
    studio = AblationStudio()
    results = studio.run_trial(simulate_evaluator, n_trials=N_TRIALS)
    
    studio.generate_recommendations()
    studio.print_report()
    
    # Export
    out_path = OUTPUT_DIR / 'ablation_report.json'
    with open(out_path, 'w') as f:
        json.dump(studio.export_for_dashboard(), f, indent=2)
    print(f'[Ablation] Reporte guardado en {out_path}')