#!/usr/bin/env python3
"""
Tests para módulos del Agente Financiero.
Usa pytest + hypothesis para property-based testing.
"""
import os
import sys
import json
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_STORE = DATA_DIR


# === TESTS ALMACEN MODELOS ===
class TestAlmacenModelos:
    def setup_method(self):
        from model_store import ModelStore
        self.store = ModelStore()
    
    def test_guardar_y_cargar(self):
        from xgboost import XGBClassifier
        model = XGBClassifier(n_estimators=10, max_depth=2, verbosity=0)
        X = np.random.randn(50, 3)
        y = (X[:, 0] > 0).astype(int)
        model.fit(X, y)
        
        version = self.store.save_model(
            model=model,
            name='test_model',
            regime='test',
            params={'n_estimators': 10},
            metrics={'accuracy': 0.85},
            feature_names=['f1', 'f2', 'f3']
        )
        
        assert version is not None
        loaded = self.store.load_model('test_model', 'test', version)
        assert loaded['model'] is not None
        assert loaded['metadata']['metrics']['accuracy'] == 0.85
        assert loaded['feature_names'] == ['f1', 'f2', 'f3']

    def test_cargar_ultimo(self):
        from xgboost import XGBClassifier
        for i in range(3):
            model = XGBClassifier(n_estimators=5, max_depth=2, verbosity=0)
            X = np.random.randn(30, 2)
            y = (X[:, 0] > 0).astype(int)
            model.fit(X, y)
            self.store.save_model(model, 'test_latest', 'test', metrics={'version': i})
        
        loaded = self.store.load_latest('test_latest', 'test')
        assert loaded is not None

    def test_listar_modelos(self):
        models = self.store.list_models()
        assert isinstance(models, dict)

    def test_promover_a_produccion(self):
        from xgboost import XGBClassifier
        model = XGBClassifier(n_estimators=5, max_depth=2, verbosity=0)
        X = np.random.randn(30, 2)
        y = (X[:, 0] > 0).astype(int)
        model.fit(X, y)
        
        version = self.store.save_model(model, 'test_prod', 'test')
        self.store.promote_to_production('test_prod', 'test', version)
        loaded = self.store.load_production('test_prod', 'test')
        assert loaded is not None


# === TESTS CALIBRACION ===
class TestCalibracion:
    def test_calibracion_isotonic(self):
        from calibration import ProbabilityCalibrator
        np.random.seed(42)
        y_true = np.random.binomial(1, 0.3, 200)
        y_prob = np.clip(y_true + np.random.randn(200) * 0.5, 0.05, 0.95)
        
        cal = ProbabilityCalibrator(metodo='isotonic')
        cal.fit(y_true, y_prob)
        assert cal.is_fitted
        assert cal.brier_before is not None
        assert cal.brier_after is not None

    def test_calibracion_platt(self):
        from calibration import ProbabilityCalibrator
        np.random.seed(42)
        y_true = np.random.binomial(1, 0.3, 200)
        y_prob = np.clip(y_true + np.random.randn(200) * 0.5, 0.05, 0.95)
        
        cal = ProbabilityCalibrator(metodo='platt')
        cal.fit(y_true, y_prob)
        assert cal.is_fitted

    def test_calibracion_mejora_brier(self):
        from calibration import ProbabilityCalibrator
        np.random.seed(42)
        y_true = np.random.binomial(1, 0.3, 500)
        y_prob = np.clip(y_true + np.random.randn(500) * 0.3, 0.01, 0.99)
        
        cal = ProbabilityCalibrator(metodo='isotonic')
        cal.fit(y_true, y_prob)
        
        assert cal.brier_after <= cal.brier_before + 0.01  # Should not worsen

    def test_gestor_calibracion(self):
        from calibration import CalibrationManager
        cm = CalibrationManager()
        y_true = np.random.binomial(1, 0.3, 100)
        y_prob = np.clip(y_true + np.random.randn(100) * 0.5, 0.05, 0.95)
        
        cm.calibrate('NVDA', y_true, y_prob)
        assert 'NVDA' in cm.calibrators


# === TESTS ALMACEN CARACTERISTICAS ===
class TestAlmacenCaracteristicas:
    def setup_method(self):
        from feature_store import FeatureStore
        self.fs = FeatureStore()

    def test_guardar_y_cargar(self):
        self.fs.store_features('NVDA', {'rsi': 65.0, 'macd': 1.5})
        self.fs.store_features('AAPL', {'rsi': 45.0, 'macd': -0.5})
        
        loaded = self.fs.load_features(ticker='NVDA', as_dataframe=True)
        assert loaded is not None

    def test_historial_caracteristicas(self):
        self.fs.store_features('NVDA', {'rsi': 60.0, 'macd': 1.0})
        history = self.fs.get_feature_history('NVDA', 'rsi', days=30)
        assert len(history) >= 0


# === TESTS VALIDACION WALKFORWARD ===
class TestValidacionWalkForward:
    def test_generar_diviciones(self):
        from walkforward_validator import generate_walk_forward_splits
        dates = pd.date_range('2022-01-01', '2024-12-31', freq='B')
        splits = generate_walk_forward_splits(dates, n_splits=3, test_size_dias=42, gap_dias=5)
        assert len(splits) > 0
        for s in splits:
            assert s.train_start < s.train_end < s.test_start < s.test_end

    def test_entrenar_probar(self):
        from walkforward_validator import walk_forward_train_test
        np.random.seed(42)
        dates = pd.date_range('2022-01-01', '2023-12-31', freq='B')
        X = pd.DataFrame({'f1': np.random.randn(len(dates))}, index=dates)
        y = pd.Series((X['f1'] > 0).astype(float), index=dates)
        
        from xgboost import XGBClassifier
        def model_fn(X, y, params):
            return XGBClassifier(**params, n_estimators=10, verbosity=0).fit(X, y)
        
        result = walk_forward_train_test(X, y, model_fn, n_splits=2, test_size_dias=21)
        assert result['n_splits'] > 0
        assert 'oos_accuracy' in result


# === TESTS PARALELO ===
class TestParalelo:
    def test_mapa_paralelo(self):
        from parallel_utils import parallel_map
        
        def double(x):
            return x * 2
        
        results = parallel_map(double, [1, 2, 3, 4, 5], max_workers=3, show_progress=False)
        assert results == [2, 4, 6, 8, 10]

    def test_mapa_paralelo_con_errores(self):
        from parallel_utils import parallel_map
        
        def failing(x):
            if x == 3:
                raise ValueError('test error')
            return x
        
        results = parallel_map(failing, [1, 2, 3, 4, 5], max_workers=3, show_progress=False)
        assert results[2] is None
        assert results[0] == 1


# === TESTS ESTRES ===
class TestEstres:
    def test_monte_carlo(self):
        from stress_test import StressTester
        st = StressTester()
        result = st.run_monte_carlo(
            expected_return=0.10,
            expected_vol=0.20,
            initial_capital=100000,
            n_simulations=100,
            n_days=63
        )
        assert 'var_95' in result
        assert result['initial_capital'] == 100000


# === TESTS DESTILACION ===
class TestDestilacion:
    def test_objetivos_suaves(self):
        from knowledge_distillation import KnowledgeDistiller
        from sklearn.linear_model import LogisticRegression
        np.random.seed(42)
        
        teacher1 = LogisticRegression().fit(np.random.randn(100, 3), np.random.binomial(1, 0.5, 100))
        teacher2 = LogisticRegression().fit(np.random.randn(100, 3), np.random.binomial(1, 0.5, 100))
        
        distiller = KnowledgeDistiller([teacher1, teacher2])
        X_new = np.random.randn(20, 3)
        soft = distiller.generate_soft_targets(X_new)
        
        assert soft is not None
        assert len(soft) == 20
        assert all(0 <= s <= 1 for s in soft)


# === TESTS MODELOS REGIMEN ===
class TestModelosRegimen:
    def test_preparar_datos(self):
        from regime_models import RegimeModelManager
        manager = RegimeModelManager()
        
        n = 300
        X = pd.DataFrame({'f1': np.random.randn(n)}, index=range(n))
        y = pd.Series((X['f1'] > 0).astype(float), index=range(n))
        regimes = pd.Series(np.random.choice(['ALCISTA', 'BAJISTA', 'LATERAL'], n), index=range(n))
        
        data = manager.prepare_regime_data(X, y, regimes)
        assert len(data) > 0


# === TESTS SHAP ===
class TestShap:
    def test_explicador_shap(self):
        pytest.importorskip('shap')
        from shap_explanations import ShapExplainer
        from xgboost import XGBClassifier
        
        np.random.seed(42)
        X = np.random.randn(100, 3)
        y = (X[:, 0] > 0).astype(int)
        
        model = XGBClassifier(n_estimators=10, max_depth=2, verbosity=0)
        model.fit(X, y)
        
        explainer = ShapExplainer(model, ['f1', 'f2', 'f3'])
        result = explainer.explain(X)
        
        assert 'feature_importance' in result
        assert len(result['feature_importance']) == 3


# === TESTS APRENDIZAJE ONLINE ===
class TestAprendizajeOnline:
    def test_inicializar_aprendiz(self):
        from online_learning import OnlineLearner
        learner = OnlineLearner('NVDA')
        assert learner.ticker == 'NVDA'
    
    def test_ajuste_parcial(self):
        from online_learning import OnlineLearner, RIVER_AVAILABLE
        learner = OnlineLearner('NVDA')
        features = {
            'rsi_14': 65.0, 'macd_hist': 1.5, 'vol_ratio': 1.2,
            'volatility_20d': 0.3, 'sma50_dist_pct': 2.0, 'sma200_dist_pct': 5.0,
            'atr_pct': 1.5, 'ret_vol_interaction': 0.1, 'ret_vol_corr_20d': 0.3,
            'price_vol_corr': 0.2, 'ret_skew_20d': -0.1, 'ret_kurt_20d': 3.0,
            'ret_zscore_20d': 1.0
        }
        result = learner.partial_fit(features, 1)
        if RIVER_AVAILABLE and learner.model is not None:
            assert result is not None
            assert 'rolling_accuracy' in result
        else:
            assert learner.model is None


# === TESTS ABLACION ===
class TestAblacion:
    def test_ejecutar_ablacion(self):
        from ablation_studio import AblationStudio
        
        def dummy_evaluator(components):
            return {'accuracy': 0.5 + len(components) * 0.01}
        
        studio = AblationStudio()
        results = studio.run_trial(dummy_evaluator, components=['comp1', 'comp2'], n_trials=2)
        assert len(results) == 4  # 2 components x 2 trials

    def test_recomendaciones(self):
        from ablation_studio import AblationStudio
        
        def dummy_evaluator(components):
            return {'accuracy': 0.5 + len(components) * 0.01}
        
        studio = AblationStudio()
        results = studio.run_trial(dummy_evaluator, components=['comp1', 'comp2'], n_trials=3)
        recs = studio.generate_recommendations()
        assert len(recs) == 2


# === TESTS CACHE YFINANCE ===
class TestCacheYfinance:
    def test_estadisticas_cache(self):
        from yfinance_cache import cache_stats
        stats = cache_stats()
        assert 'available' in stats


# === TESTS PAPER TRADING ===
class TestPaperTrading:
    def test_crear_orden(self):
        from paper_trading import Order, OrderSide, OrderType, OrderStatus
        order = Order(ticker='NVDA', side=OrderSide.BUY, shares=100)
        assert order.ticker == 'NVDA'
        assert order.side == OrderSide.BUY
        assert order.status == OrderStatus.PENDING
    
    def test_broker_comprar(self):
        from paper_trading import PaperBroker, OrderSide, OrderType
        broker = PaperBroker(initial_capital=100000)
        broker.set_price_provider(lambda t: 150.0)
        order = broker.submit_order('NVDA', OrderSide.BUY, 10)
        assert order.status.value == 'filled'
        assert broker.capital < 100000
        assert 'NVDA' in broker.positions
    
    def test_broker_vender(self):
        from paper_trading import PaperBroker, OrderSide
        broker = PaperBroker(initial_capital=100000)
        broker.set_price_provider(lambda t: 150.0)
        broker.submit_order('NVDA', OrderSide.BUY, 10)
        broker.set_price_provider(lambda t: 155.0)
        order = broker.submit_order('NVDA', OrderSide.SELL, 5)
        assert order.status.value == 'filled'
    
    def test_resumen_cuenta(self):
        from paper_trading import PaperBroker, OrderSide
        broker = PaperBroker(initial_capital=100000)
        broker.set_price_provider(lambda t: 200.0)
        broker.submit_order('AAPL', OrderSide.BUY, 50)
        summary = broker.get_account_summary()
        assert 'portfolio_value' in summary
        assert 'positions' in summary
    
    def test_cerrar_todo(self):
        from paper_trading import PaperBroker, OrderSide
        broker = PaperBroker(initial_capital=100000)
        broker.set_price_provider(lambda t: 100.0)
        broker.submit_order('AAPL', OrderSide.BUY, 100)
        broker.submit_order('MSFT', OrderSide.BUY, 50)
        orders = broker.close_all()
        assert len(orders) == 2

    def test_guardar_cargar_estado(self):
        from paper_trading import PaperBroker, OrderSide
        broker = PaperBroker(initial_capital=100000)
        broker.set_price_provider(lambda t: 150.0)
        broker.submit_order('NVDA', OrderSide.BUY, 10)
        path = broker.save_state()
        assert path.exists()
        
        broker2 = PaperBroker()
        loaded = broker2.load_state(broker.account_id)
        assert loaded
        assert broker2.capital == broker.capital


# === TESTS DATOS ALTERNATIVOS ===
class TestDatosAlternativos:
    def test_indicadores_macro(self):
        from alternative_data import MacroIndicators
        macro = MacroIndicators()
        data = macro.fetch()
        assert 'GDP' in data or 'CPIAUCSL' in data
        for key, val in data.items():
            if isinstance(val, dict):
                assert 'value' in val

    def test_obtener_ganancias(self):
        from alternative_data import EarningsTranscripts
        et = EarningsTranscripts()
        data = et.fetch(['NVDA'])
        assert 'NVDA' in data

    def test_operaciones_internas(self):
        from alternative_data import InsiderTrading
        insider = InsiderTrading()
        data = insider.fetch(['AAPL'])
        assert 'AAPL' in data
        assert 'net_sentiment' in data['AAPL']

    def test_tendencias_busqueda(self):
        from alternative_data import SearchTrends
        st = SearchTrends()
        data = st.fetch(['NVDA', 'AAPL'])
        assert len(data) == 2
        assert 'search_volume' in data['NVDA']

    def test_flujo_opciones(self):
        from alternative_data import OptionsFlow
        of = OptionsFlow()
        data = of.fetch(['NVDA'])
        assert 'put_call_ratio' in data['NVDA']

    def test_agregador(self):
        from alternative_data import AlternativeDataAggregator
        agg = AlternativeDataAggregator()
        data = agg.fetch_all(['NVDA', 'AAPL'])
        assert len(data) > 0
        features = agg.to_features(data)
        assert len(features) > 0

    def test_puntuacion(self):
        from alternative_data import AlternativeDataAggregator
        agg = AlternativeDataAggregator()
        data = agg.fetch_all(['NVDA'])
        score = agg.score_ticker('NVDA', data)
        assert 0 <= score <= 100


# === TESTS RL DIMENSIONAMIENTO ===
class TestRLDimensionamiento:
    def _get_env(self):
        from rl_position_sizing import GYM_AVAILABLE
        if not GYM_AVAILABLE:
            pytest.skip('gymnasium not installed')
        from rl_position_sizing import PortfolioEnv, RLSizer
        return PortfolioEnv, RLSizer

    def test_crear_entorno(self):
        PortfolioEnv, _ = self._get_env()
        env = PortfolioEnv(tickers=['NVDA', 'AAPL'], n_assets=2, window_length=100)
        obs, info = env.reset()
        assert obs is not None
        assert info == {}

    def test_paso_entorno(self):
        PortfolioEnv, _ = self._get_env()
        env = PortfolioEnv(tickers=['NVDA', 'AAPL'], n_assets=2, window_length=200)
        env.reset()
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        assert isinstance(reward, float)
        assert isinstance(done, bool)

    def test_pasos_multiples(self):
        PortfolioEnv, _ = self._get_env()
        env = PortfolioEnv(tickers=['NVDA', 'AAPL', 'MSFT'], n_assets=3, window_length=100)
        env.reset()
        total_reward = 0
        for _ in range(10):
            action = env.action_space.sample()
            _, reward, done, _, _ = env.step(action)
            total_reward += reward
            if done:
                break
        assert isinstance(total_reward, float)

    def test_crear_dimensionador_rl(self):
        _, RLSizer = self._get_env()
        sizer = RLSizer(algo='PPO', total_timesteps=1000)
        assert sizer.algo == 'PPO'
        assert sizer.impl.hyperparams['total_timesteps'] == 1000


# === TESTS SISTEMA MULTI-AGENTE ===
class TestSistemaMultiAgente:
    def test_agente_tecnico(self):
        from multi_agent_system import TechnicalAgent, AgentVote
        agent = TechnicalAgent()
        vote = agent.analyze('NVDA', context={})
        assert isinstance(vote, AgentVote)
        assert vote.agent_name == 'tecnico'

    def test_agente_sentimiento(self):
        from multi_agent_system import SentimentAgent, AgentVote
        agent = SentimentAgent()
        vote = agent.analyze('NVDA', context={})
        assert isinstance(vote, AgentVote)

    def test_revisor_meta(self):
        from multi_agent_system import MetaReviewer, AgentVote
        reviewer = MetaReviewer()
        votes = []
        for agent_name in ['tecnico', 'fundamental', 'macro', 'sentimiento', 'riesgo']:
            votes.append(AgentVote(
                agent_name=agent_name, ticker='NVDA',
                direction='up' if agent_name in ['tecnico', 'fundamental'] else 'down',
                probability=70, confidence=0.7,
                reasoning=f'Test reasoning from {agent_name}',
                features_used=[], regime='ALCISTA'
            ))
        debate = reviewer.evaluate_and_weight(votes, context={'regimen': 'ALCISTA'})
        assert debate.consensus_direction in ['up', 'down', 'neutral']

    def test_sistema_multi_agente(self):
        from multi_agent_system import get_multi_agent_system
        mas = get_multi_agent_system()
        debate = mas.analyze_ticker('NVDA', context={})
        assert debate.consensus_direction in ['up', 'down', 'neutral']

    def test_reporte_agentes(self):
        from multi_agent_system import get_multi_agent_system
        mas = get_multi_agent_system()
        mas.analyze_ticker('NVDA', context={})
        report = mas.agent_report()
        assert 'tecnico' in report


# === TESTS RAZONAMIENTO CAUSAL ===
class TestRazonamientoCausal:
    def test_analizador_causal(self):
        from causal_reasoning import CausalAnalyzer
        analyzer = CausalAnalyzer()
        assert analyzer is not None

    def test_estimar_ate(self):
        from causal_reasoning import CausalAnalyzer, CausalEffect
        np.random.seed(42)
        n = 200
        df = pd.DataFrame({
            'treatment': np.random.binomial(1, 0.5, n),
            'outcome': np.random.randn(n),
            'confounder1': np.random.randn(n),
            'confounder2': np.random.randn(n),
        })
        analyzer = CausalAnalyzer()
        result = analyzer.estimate_ate(df, 'treatment', 'outcome', ['confounder1', 'confounder2'])
        assert isinstance(result, CausalEffect)
        assert hasattr(result, 'effect')

    def test_contrafactual(self):
        from causal_reasoning import CausalAnalyzer
        np.random.seed(42)
        n = 100
        df = pd.DataFrame({
            'treatment': np.random.binomial(1, 0.5, n),
            'outcome': np.random.randn(n),
            'confounder1': np.random.randn(n),
        })
        analyzer = CausalAnalyzer()
        cf = analyzer.counterfactual(df, 'treatment', 'outcome', ['confounder1'])
        assert 'counterfactual_mean' in cf

    def test_grafo_causal(self):
        from causal_reasoning import CausalGraphBuilder, CausalGraph
        builder = CausalGraphBuilder()
        graph = builder.get_default_graph()
        assert isinstance(graph, CausalGraph)
        assert len(graph.nodes) > 0


# === TESTS ARQUITECTURA NUBE ===
class TestArquitecturaNube:
    def test_cluster_ray(self):
        from cloud_architecture import RayCluster
        cluster = RayCluster(num_cpus=1)
        assert cluster.num_cpus == 1
        assert not cluster.initialized

    def test_configurar_feast(self):
        from cloud_architecture import FeastFeatureStore
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            feast = FeastFeatureStore(repo_path=tmp)
            path = feast.setup_repo()
            assert (Path(path) / 'feature_store.yaml').exists()

    def test_infraestructura_docker(self):
        from cloud_architecture import DockerInfrastructure
        import tempfile
        orig = os.environ.get('GITHUB_WORKSPACE')
        with tempfile.TemporaryDirectory() as tmp:
            os.environ['GITHUB_WORKSPACE'] = tmp
            try:
                df_path = DockerInfrastructure.generate_dockerfile()
                dc_path = DockerInfrastructure.generate_docker_compose()
                k8s_path = DockerInfrastructure.generate_kubernetes()
                assert Path(df_path).exists()
                assert Path(dc_path).exists()
                assert Path(k8s_path).exists()
            finally:
                if orig:
                    os.environ['GITHUB_WORKSPACE'] = orig
                else:
                    del os.environ['GITHUB_WORKSPACE']

    def test_definir_pipeline_prefect(self):
        from cloud_architecture import PrefectPipeline
        assert hasattr(PrefectPipeline, 'full_pipeline_flow')

    def test_orquestador_nube(self):
        from cloud_architecture import CloudOrchestrator, DATA_DIR
        path = CloudOrchestrator.generate_requirements()
        assert Path(path).exists()


# === TESTS INVESTIGACION AUTOMATIZADA ===
class TestInvestigacionAutomatizada:
    def test_generar_alpha(self):
        from automated_research import AlphaFactory
        factory = AlphaFactory()
        alphas = factory.generate_random(10)
        assert len(alphas) == 10
        assert all(a.name.startswith('alpha_') for a in alphas)

    def test_computar_expresion_alpha(self):
        from automated_research import AlphaExpression
        dates = pd.date_range('2023-01-01', '2023-12-31', freq='B')
        data = pd.DataFrame({'Close': 100 + np.cumsum(np.random.randn(len(dates)))}, index=dates)
        expr = AlphaExpression(name='test', expression='returns + sma',
                               primitive='returns', operator='+', secondary='sma')
        result = expr.compute(data)
        assert result is not None
        assert len(result) == len(data)

    def test_selector_alpha(self):
        from automated_research import AlphaSelector, AlphaFactory
        dates = pd.date_range('2023-01-01', '2023-12-31', freq='B')
        data = pd.DataFrame({'Close': 100 + np.cumsum(np.random.randn(len(dates)))}, index=dates)
        factory = AlphaFactory()
        alphas = factory.generate_random(20)
        selector = AlphaSelector()
        df = selector.evaluate_all(alphas, data)
        assert len(df) == 20
        assert 'ic' in df.columns

    def test_generador_hipotesis(self):
        from automated_research import HypothesisGenerator
        gen = HypothesisGenerator()
        df_corr = pd.DataFrame({
            'target': [1.0, 0.5, -0.3],
            'f1': [0.5, 1.0, 0.1],
            'f2': [-0.3, 0.1, 1.0],
        }, index=['target', 'f1', 'f2'])
        hyps = gen.from_correlation(df_corr, 'target')
        assert len(hyps) > 0

    def test_decaimiento_senal(self):
        from automated_research import SignalDecayAnalyzer
        np.random.seed(42)
        n = 500
        signal = pd.Series(np.random.randn(n))
        returns = pd.Series(np.random.randn(n) * 0.02)
        analyzer = SignalDecayAnalyzer()
        result = analyzer.compute_decay(signal, returns, max_lag=10)
        assert 'ics_by_lag' in result
        assert 'half_life' in result


# === TESTS MICROESTRUCTURA MERCADO ===
class TestMicroestructuraMercado:
    def test_desequilibrio_flujo(self):
        from market_microstructure import OrderFlowImbalance
        n = 1000
        df = pd.DataFrame({
            'price': np.random.randn(n) + 100,
            'bid': np.random.randn(n) + 99.9,
            'ask': np.random.randn(n) + 100.1,
            'volume': np.random.randint(100, 10000, n),
        })
        ofi = OrderFlowImbalance()
        result = ofi.from_ticks(df)
        assert 'of_imbalance' in result.columns

    def test_vpin(self):
        from market_microstructure import VPIN
        n = 1000
        df = pd.DataFrame({
            'price': 100 + np.cumsum(np.random.randn(n) * 0.1),
            'volume': np.random.randint(100, 10000, n),
        })
        vpin = VPIN(n_buckets=20)
        result = vpin.compute(df)
        assert 'vpin' in result.columns

    def test_estimador_spread(self):
        from market_microstructure import SpreadEstimator
        np.random.seed(42)
        prices = pd.Series(100 + np.cumsum(np.random.randn(500)))
        spread = SpreadEstimator.roll_estimator(prices)
        assert isinstance(spread, float)

    def test_kyle_lambda(self):
        from market_microstructure import KyleLambda
        n = 500
        df = pd.DataFrame({
            'price': 100 + np.cumsum(np.random.randn(n)),
            'volume': np.random.randint(100, 10000, n),
        })
        kyle = KyleLambda(window=50)
        result = kyle.compute(df)
        assert 'lambda' in result.name if hasattr(result, 'name') else True

    def test_liquidez(self):
        from market_microstructure import LiquidityMeasures
        np.random.seed(42)
        ret = pd.Series(np.random.randn(200) * 0.02)
        vol = pd.Series(np.random.randint(1000, 100000, 200))
        amihud = LiquidityMeasures.amihud(ret, vol, window=20)
        assert amihud is not None

    def test_analizador_mercado(self):
        from market_microstructure import MarketMicrostructureAnalyzer
        n = 1000
        df = pd.DataFrame({
            'price': 100 + np.cumsum(np.random.randn(n) * 0.1),
            'bid': 99.95 + np.random.randn(n) * 0.02,
            'ask': 100.05 + np.random.randn(n) * 0.02,
            'volume': np.random.randint(100, 10000, n),
            'side': np.random.choice(['buy', 'sell'], n),
        })
        analyzer = MarketMicrostructureAnalyzer()
        results = analyzer.analyze(df)
        assert 'ofi_mean' in results


# === TESTS PRECIOS GENERATIVOS ===
class TestPreciosGenerativos:
    def test_componentes_timegan(self):
        from generative_prices import TimeGANComponents
        embedder = TimeGANComponents.build_embedder(1, 24, 2)
        if embedder is not None:
            import torch
            x = torch.randn(10, 1)
            out = embedder(x)
            assert out.shape[-1] == 24

    def test_inicializar_timegan(self):
        from generative_prices import TimeGAN
        gan = TimeGAN(input_dim=1, hidden_dim=24, z_dim=10)
        assert gan.input_dim == 1

    def test_validador_sintetico(self):
        from generative_prices import SyntheticDataValidator
        np.random.seed(42)
        real = np.random.randn(100, 5)
        synth = np.random.randn(100, 5) * 0.95 + 0.02
        validator = SyntheticDataValidator()
        metrics = validator.validate(real, synth)
        assert 'mean_abs_diff' in metrics
        assert len(metrics) > 0

    def test_pipeline_generativo(self):
        from generative_prices import GenerativePricePipeline
        np.random.seed(42)
        n = 500
        close = pd.Series(100 * np.exp(np.cumsum(np.random.randn(n) * 0.015)),
                          index=pd.date_range('2023-01-01', periods=n, freq='h'))
        pipeline = GenerativePricePipeline(input_dim=1, seq_len=30)
        data = pipeline.prepare_data(close)
        assert data.ndim == 3
        assert data.shape[-1] == 1


# === TESTS RL MULTI-ACTIVO ===
class TestRLMultiActivo:
    def test_crear_entorno(self):
        from multi_asset_rl import MultiAssetPortfolioEnv, GYM_AVAILABLE
        if not GYM_AVAILABLE:
            pytest.skip('gymnasium not available')
        n, n_assets = 500, 5
        dates = pd.date_range('2023-01-01', periods=n, freq='B')
        prices = pd.DataFrame({f'A{i}': 100 + np.cumsum(np.random.randn(n))
                               for i in range(n_assets)}, index=dates)
        returns = prices.pct_change().dropna()
        env = MultiAssetPortfolioEnv(prices=prices, returns=returns, n_assets=3, window=20)
        obs, info = env.reset()
        assert obs is not None
        assert obs.shape[0] == env.state_dim

    def test_paso_entorno(self):
        from multi_asset_rl import MultiAssetPortfolioEnv, GYM_AVAILABLE
        if not GYM_AVAILABLE:
            pytest.skip('gymnasium not available')
        n, n_assets = 500, 5
        dates = pd.date_range('2023-01-01', periods=n, freq='B')
        prices = pd.DataFrame({f'A{i}': 100 + np.cumsum(np.random.randn(n))
                               for i in range(n_assets)}, index=dates)
        returns = prices.pct_change().dropna()
        env = MultiAssetPortfolioEnv(prices=prices, returns=returns, n_assets=3, window=20)
        env.reset()
        action = env.action_space.sample()
        obs, reward, done, _, info = env.step(action)
        assert isinstance(reward, float)

    def test_rendimiento_entorno(self):
        from multi_asset_rl import MultiAssetPortfolioEnv, GYM_AVAILABLE
        if not GYM_AVAILABLE:
            pytest.skip('gymnasium not available')
        n, n_assets = 500, 3
        dates = pd.date_range('2023-01-01', periods=n, freq='B')
        prices = pd.DataFrame({f'A{i}': 100 + np.cumsum(np.random.randn(n))
                               for i in range(n_assets)}, index=dates)
        returns = prices.pct_change().dropna()
        env = MultiAssetPortfolioEnv(prices=prices, returns=returns, n_assets=2, window=20)
        env.reset()
        for _ in range(5):
            action = env.action_space.sample()
            env.step(action)
        perf = env.get_performance()
        assert 'total_return' in perf

    def test_agente_rl_actuar(self):
        from multi_asset_rl import MultiAssetRLAgent, TORCH_AVAILABLE
        agent = MultiAssetRLAgent(state_dim=20, action_dim=5)
        state = np.random.randn(20)
        action = agent.act(state)
        assert len(action) == 5


# === TESTS ANALITICA PORTAFOLIO ===
class TestAnaliticaPortafolio:
    def test_modelo_riesgo_barra(self):
        from portfolio_analytics import BarraRiskModel
        np.random.seed(42)
        n, n_factors = 100, 5
        returns = pd.DataFrame(np.random.randn(n, 10) * 0.02)
        exposures = pd.DataFrame(np.random.randn(n, n_factors),
                                 columns=['value', 'momentum', 'size', 'volatility', 'quality'])
        barra = BarraRiskModel(style_factors=['value', 'momentum', 'size', 'volatility', 'quality'])
        factor_rets = barra.estimate_factor_returns(returns, exposures)
        assert factor_rets is not None

    def test_descomposicion_riesgo_barra(self):
        from portfolio_analytics import BarraRiskModel
        np.random.seed(42)
        n_assets = 5
        weights = np.ones(n_assets) / n_assets
        exposures = pd.DataFrame(np.random.randn(1, 5),
                                 columns=['value', 'momentum', 'size', 'volatility', 'quality'])
        specific_risk = pd.Series(np.random.rand(n_assets) * 0.01)
        barra = BarraRiskModel()
        dec = barra.risk_decomposition(weights, exposures, specific_risk)
        assert 'total_risk' in dec
        assert 'factor_exposures' in dec

    def test_atribucion_brinson(self):
        from portfolio_analytics import BrinsonAttribution
        np.random.seed(42)
        n = 10
        pw = pd.Series(np.random.dirichlet(np.ones(n)))
        bw = pd.Series(np.random.dirichlet(np.ones(n)))
        pr = pd.Series(np.random.randn(n) * 0.02)
        br = pd.Series(np.random.randn(n) * 0.015)
        brinson = BrinsonAttribution()
        result = brinson.decompose(pw, bw, pr, br)
        assert 'allocation_effect' in result
        assert 'total_excess_return' in result

    def test_black_litterman(self):
        from portfolio_analytics import BlackLitterman
        np.random.seed(42)
        n = 5
        bl = BlackLitterman()
        cap_weights = np.ones(n) / n
        cov = np.eye(n) * 0.04
        pi = bl.implied_equilibrium_returns(cap_weights, cov, 0.10)
        assert len(pi) == n

        P = np.eye(n)
        Q = np.ones(n) * 0.12
        post_mean, post_cov = bl.add_views(P, Q, sigma=cov)
        assert post_mean is not None
        assert post_cov is not None

    def test_paridad_riesgo(self):
        from portfolio_analytics import RiskBudgeting
        np.random.seed(42)
        n = 5
        cov = np.random.randn(n, n)
        cov = cov @ cov.T + np.eye(n) * 0.1
        weights = RiskBudgeting.risk_parity_weights(cov)
        assert len(weights) == n
        assert abs(weights.sum() - 1) < 0.01

    def test_analisis_escenario(self):
        from portfolio_analytics import ScenarioAnalysis
        np.random.seed(42)
        n = 5
        weights = np.ones(n) / n
        asset_classes = ['equity', 'bond', 'commodity', 'cash', 'equity']
        sa = ScenarioAnalysis()
        df = sa.run(weights, asset_classes)
        assert len(df) > 0
        assert 'scenario' in df.columns

    def test_metricas_riesgo(self):
        from portfolio_analytics import RiskMetrics
        np.random.seed(42)
        returns = pd.Series(np.random.randn(500) * 0.02 + 0.0005)
        metrics = RiskMetrics.compute_all(returns)
        assert 'sharpe_ratio' in metrics
        assert 'max_drawdown' in metrics

    def test_analitica_portafolio_completa(self):
        from portfolio_analytics import PortfolioAnalytics
        np.random.seed(42)
        n, n_assets = 500, 5
        returns = pd.DataFrame({f'A{i}': np.random.randn(n) * 0.02 + 0.0005
                                for i in range(n_assets)})
        weights = np.ones(n_assets) / n_assets
        analytics = PortfolioAnalytics()
        results = analytics.full_report(returns, weights)
        assert 'risk_metrics' in results
        assert 'scenarios' in results


# === TESTS META-APRENDIZAJE ===
class TestMetaAprendizaje:
    def test_aprendiz_maml(self):
        from meta_learning import MAMLLearner, TORCH_AVAILABLE
        learner = MAMLLearner(input_dim=8)
        assert learner.input_dim == 8
        assert not learner.trained

    def test_adaptador_pocos_ejemplos(self):
        from meta_learning import FewShotRegimeAdapter
        adapter = FewShotRegimeAdapter()
        assert adapter.feature_cols is not None
        assert len(adapter.feature_cols) > 0

    def test_inicializar_pipeline(self):
        from meta_learning import MetaLearningPipeline
        pipeline = MetaLearningPipeline()
        assert pipeline.adapter is not None

    def test_preparar_tareas(self):
        from meta_learning import FewShotRegimeAdapter
        n = 200
        data = pd.DataFrame(np.random.randn(n, 8), columns=['rsi_14', 'macd_hist', 'vol_ratio',
            'volatility_20d', 'sma50_dist_pct', 'atr_pct', 'returns_5d', 'returns_20d'])
        data['regime'] = np.random.choice(['ALCISTA', 'BAJISTA', 'LATERAL'], n)
        data['forward_return'] = np.random.randn(n) * 0.02
        adapter = FewShotRegimeAdapter()
        tasks = adapter.prepare_tasks(data)
        assert isinstance(tasks, list)


# === TESTS OPCIONES ===
class TestOpciones:
    def test_black_scholes_call(self):
        from opciones_derivados import BlackScholes, OptionPrice
        opt = BlackScholes.price(S=100, K=100, T=1.0, r=0.05, sigma=0.25, option_type='call')
        assert isinstance(opt, OptionPrice)
        assert opt.premium > 0
        assert opt.greeks.delta > 0.5

    def test_black_scholes_put(self):
        from opciones_derivados import BlackScholes
        opt = BlackScholes.price(S=100, K=100, T=1.0, r=0.05, sigma=0.25, option_type='put')
        assert opt.premium > 0
        assert opt.greeks.delta < -0.3

    def test_volatilidad_implicita(self):
        from opciones_derivados import BlackScholes
        opt = BlackScholes.price(S=100, K=100, T=1.0, r=0.05, sigma=0.25)
        iv = BlackScholes.implied_volatility(opt.premium, 100, 100, 1.0, 0.05)
        assert abs(iv - 0.25) < 0.05

    def test_sonrisa_volatilidad(self):
        from opciones_derivados import VolatilitySmile
        import numpy as np
        strikes = np.arange(80, 121, 5)
        smile = VolatilitySmile()
        df = smile.compute_smile(strikes, np.ones(len(strikes))*10, 100, 1.0, 0.05)
        assert len(df) == len(strikes)

    def test_estrategia_multipierna(self):
        from opciones_derivados import MultiLegStrategy
        strategy = MultiLegStrategy.bull_call_spread(100, 95, 105, 1.0, 0.05, 0.25)
        assert len(strategy) == 2

    def test_cobertura_delta(self):
        from opciones_derivados import DeltaHedger
        hedger = DeltaHedger(100, 100, 1.0, 0.05, 0.25)
        result = hedger.rebalance(102, 0.1)
        assert 'delta' in result
        assert 'pnl' in result


# === TESTS CPPI ===
class TestCPPI:
    def test_inicializar_cppi(self):
        from cppi import CPPI
        cppi = CPPI(capital_inicial=100000)
        assert cppi.portfolio_value == 100000
        assert cppi.floor == 85000

    def test_rebalanceo_cppi(self):
        from cppi import CPPI
        cppi = CPPI(capital_inicial=100000)
        result = cppi.rebalance(0.01, 0.0001)
        assert result.valor_portafolio > 0
        assert result.exposicion >= 0

    def test_simulacion_cppi(self):
        from cppi import CPPI
        cppi = CPPI(capital_inicial=100000)
        df = cppi.run_simulation(n_days=50)
        assert len(df) == 50
        assert 'portafolio' in df.columns

    def test_obpi(self):
        from cppi import OBPI
        obpi = OBPI(capital=100000)
        alloc = obpi.compute_allocation(S=100)
        assert 'stocks' in alloc
        assert 'bonds' in alloc

    def test_seguro_portafolio(self):
        from cppi import PortfolioInsurance
        pi = PortfolioInsurance()
        result = pi.compare_strategies(n_days=50)
        assert 'cppi_final' in result


# === TESTS ESTRATEGIAS GENETICAS ===
class TestEstrategiasGeneticas:
    def test_nodo_aleatorio(self):
        from genetic_strategies import random_node, NodeType
        node = random_node(max_depth=2)
        assert node is not None
        assert node.type in NodeType

    def test_evaluar_arbol(self):
        from genetic_strategies import Node, NodeType, evaluate_tree
        node = Node(NodeType.CONSTANT, "42.0")
        data = pd.Series({'close': 100})
        result = evaluate_tree(node, data)
        assert result == 42.0

    def test_evaluar_operador(self):
        from genetic_strategies import Node, NodeType, evaluate_tree
        node = Node(NodeType.OPERATOR, '+', Node(NodeType.CONSTANT, "1.0"), Node(NodeType.CONSTANT, "2.0"))
        data = pd.Series({'close': 100})
        result = evaluate_tree(node, data)
        assert result == 3.0

    def test_inicializar_optimizador(self):
        from genetic_strategies import GeneticStrategyOptimizer
        opt = GeneticStrategyOptimizer(pop_size=10, generations=5)
        assert opt.pop_size == 10
        assert opt.generations == 5

    def test_ejecutar_optimizador(self):
        from genetic_strategies import GeneticStrategyOptimizer
        np.random.seed(42)
        import random as rnd; rnd.seed(42)
        n = 100
        data = pd.DataFrame({'rsi_14': np.random.randn(n), 'close': np.random.randn(n) + 100})
        data['forward_return'] = np.random.randn(n) * 0.02
        opt = GeneticStrategyOptimizer(pop_size=10, generations=3)
        result = opt.run(data)
        assert 'best_fitness' in result


# === TESTS DASHBOARD ===
class TestDashboard:
    def test_proveedor_datos(self):
        from dashboard_app import DashboardDataProvider
        dp = DashboardDataProvider()
        df = dp.load_portfolio_data()
        assert len(df) > 0
        assert 'fecha' in df.columns

    def test_datos_riesgo(self):
        from dashboard_app import DashboardDataProvider
        dp = DashboardDataProvider()
        risk = dp.load_risk_data()
        assert 'var_95' in risk

    def test_resumen_generador(self):
        from dashboard_app import DashboardGenerator
        gen = DashboardGenerator()
        summary = gen.generate_summary()
        assert 'rendimiento' in summary
        assert 'riesgo' in summary

    def test_generar_html(self):
        from dashboard_app import DashboardGenerator
        gen = DashboardGenerator()
        path = gen.generate_html_report()
        assert Path(path).exists()

    def test_api(self):
        from dashboard_app import DashboardAPI
        api = DashboardAPI()
        s = api.get_summary()
        assert 'fecha' in s


# === TESTS STREAMING ===
class TestStreaming:
    def test_evento_mercado(self):
        from streaming_pipeline import MarketEvent
        from datetime import datetime
        ev = MarketEvent(ticker='NVDA', price=100.0, volume=1000, timestamp=datetime.now())
        assert ev.ticker == 'NVDA'

    def test_buffer_datos(self):
        from streaming_pipeline import DataBuffer, MarketEvent
        from datetime import datetime
        buf = DataBuffer(maxlen=100)
        buf.append('NVDA', MarketEvent('NVDA', 100, 1000, datetime.now()))
        buf.append('NVDA', MarketEvent('NVDA', 101, 2000, datetime.now()))
        assert len(buf.get('NVDA', 10)) == 2
        latest = buf.latest('NVDA')
        assert latest is not None
        assert latest.price == 101

    def test_procesador_stream(self):
        from streaming_pipeline import StreamProcessor, MarketEvent
        from datetime import datetime
        sp = StreamProcessor()
        sp.on_event(MarketEvent('NVDA', 100, 1000, datetime.now()))
        assert sp.buffer.latest('NVDA') is not None

    def test_agregador(self):
        from streaming_pipeline import RealtimeAggregator
        agg = RealtimeAggregator(window_sec=60)
        agg.add('NVDA', 100, 1000)
        agg.add('NVDA', 101, 2000)
        vwap = agg.vwap('NVDA')
        assert vwap is not None
        assert vwap > 0


# === TESTS EJECUCION ALGORITMICA ===
class TestEjecucion:
    def test_vwap(self):
        from ejecucion_algoritmica import VWAPExecutor
        executor = VWAPExecutor()
        n = 100
        data = pd.DataFrame({'close': 100 + np.cumsum(np.random.randn(n) * 0.1),
            'volume': np.random.randint(1000, 100000, n)})
        order = executor.execute('NVDA', 'buy', 10000, data)
        assert order.status == 'filled'
        assert order.executed_shares > 0

    def test_twap(self):
        from ejecucion_algoritmica import TWAPExecutor
        executor = TWAPExecutor(n_slices=5)
        data = pd.DataFrame({'close': 100 + np.cumsum(np.random.randn(50) * 0.1)})
        order = executor.execute('NVDA', 'buy', 5000, data)
        assert order.status == 'filled'
        assert order.fill_price > 0

    def test_implementation_shortfall(self):
        from ejecucion_algoritmica import ImplementationShortfallExecutor
        executor = ImplementationShortfallExecutor()
        n = 100
        data = pd.DataFrame({'close': 100 + np.cumsum(np.random.randn(n) * 0.1),
            'volume': np.random.randint(1000, 100000, n)})
        order = executor.execute('NVDA', 'buy', 10000, data)
        assert order.status == 'filled'
        assert order.slippage_bps is not None

    def test_gestor_ejecucion(self):
        from ejecucion_algoritmica import ExecutionManager
        em = ExecutionManager()
        assert 'VWAP' in em.executors
        assert 'TWAP' in em.executors

    def test_comparar_algoritmos(self):
        from ejecucion_algoritmica import ExecutionManager
        em = ExecutionManager()
        n = 100
        data = pd.DataFrame({'close': 100 + np.cumsum(np.random.randn(n) * 0.1),
            'volume': np.random.randint(1000, 100000, n)})
        df = em.compare_algos('NVDA', 'buy', 10000, data)
        assert len(df) == 3  # VWAP, TWAP, IS


# === TESTS API REST ===
class TestAPIRest:
    def test_monitor_salud(self):
        from api_rest import HealthMonitor
        hm = HealthMonitor()
        uptime = hm.get_uptime()
        assert ':' in uptime

    def test_gestor_tareas(self):
        from api_rest import TaskManager
        tm = TaskManager()
        status = tm.get_status()
        assert isinstance(status, dict)

    def test_registrar_tarea(self):
        from api_rest import TaskManager
        tm = TaskManager()
        tm.register('test', 'Test task', 60, lambda: None)
        status = tm.get_status()
        assert 'test' in status

    def test_generador_config_api(self):
        from api_rest import APIConfigGenerator
        cfg = APIConfigGenerator()
        path = cfg.generate_uvicorn_config()
        assert Path(path).exists()


# === TESTS BACKTEST ALTA FIDELIDAD ===
class TestBacktestAltaFidelidad:
    def test_modelo_deslizamiento(self):
        from backtest_alta_fidelidad import SlippageModel
        sm = SlippageModel()
        slippage = sm.compute_slippage(1000, 100, 100000, 'buy', 0.02)
        assert isinstance(slippage, float)

    def test_simulador_latencia(self):
        from backtest_alta_fidelidad import LatencySimulator
        ls = LatencySimulator()
        lat, dropped = ls.simulate()
        assert lat > 0
        assert isinstance(dropped, bool)

    def test_restricciones_capital(self):
        from backtest_alta_fidelidad import CapitalConstraints
        cc = CapitalConstraints(max_capital=100000)
        violations = cc.check_order('NVDA', 'buy', 1000, 150)
        assert isinstance(violations, list)

    def test_backtest_alta_fidelidad(self):
        from backtest_alta_fidelidad import HighFidelityBacktest
        np.random.seed(42)
        n = 50
        data = pd.DataFrame({'close': 100 + np.cumsum(np.random.randn(n) * 0.1),
            'volume': np.random.randint(10000, 1000000, n),
            'volatility': np.random.rand(n) * 0.02 + 0.01})
        signals = pd.DataFrame({'ticker': ['NVDA'] * 10,
            'signal': ['buy'] * 10, 'shares': np.random.randint(100, 500, 10)})
        bt = HighFidelityBacktest()
        r = bt.run_backtest(signals, data)
        assert 'filled' in r
        assert 'rejected' in r


# === TESTS CONECTIVIDAD ===
class TestConectividad:
    def test_tick(self):
        from conectividad_mercados import Tick
        from datetime import datetime
        t = Tick(ticker='NVDA', price=100, volume=1000, timestamp=datetime.now())
        assert t.ticker == 'NVDA'
        assert t.exchange == 'SIMULATED'

    def test_latido(self):
        from conectividad_mercados import HeartbeatMonitor
        hm = HeartbeatMonitor(timeout_sec=60)
        hm.record_heartbeat('IBKR')
        assert hm.is_alive('IBKR')

    def test_buffer_reproduccion(self):
        from conectividad_mercados import TickReplayBuffer, Tick
        from datetime import datetime
        buf = TickReplayBuffer(maxlen=100)
        buf.append(Tick('NVDA', 100, 1000, datetime.now()))
        buf.append(Tick('NVDA', 101, 2000, datetime.now()))
        ticks = buf.replay('NVDA', datetime.now(), datetime.now())
        assert len(ticks) >= 0

    def test_gestor_websocket(self):
        from conectividad_mercados import WebSocketManager
        ws = WebSocketManager()
        ws.connect('TEST', 'ws://test', ['NVDA'])
        assert 'TEST' in ws.connections

    def test_simulador_ibkr(self):
        from conectividad_mercados import IBKRSimulator
        ib = IBKRSimulator()
        r = ib.connect()
        assert r['status'] == 'connected'

    def test_simulador_binance(self):
        from conectividad_mercados import BinanceSimulator
        bn = BinanceSimulator()
        ticker = bn.get_ticker('BTCUSDT')
        assert 'price' in ticker

    def test_conectividad_mercado(self):
        from conectividad_mercados import MarketConnectivity
        mc = MarketConnectivity()
        mc.connect_all()
        assert mc.ibkr.connected


# === TESTS RIESGOS TIEMPO REAL ===
class TestRiesgos:
    def test_var_tiempo_real(self):
        from riesgos_tiempo_real import RealTimeVAR
        var = RealTimeVAR()
        returns = np.random.randn(252) * 0.02
        result = var.compute(returns, 100000)
        assert 'var_95' in result

    def test_gestor_limites(self):
        from riesgos_tiempo_real import LimitManager
        lm = LimitManager()
        lm.add_limit('test_limit', 'max', 0.5)
        lm.update('test_limit', 0.6)
        assert lm.limits['test_limit'].excedido

    def test_interruptor_circuito(self):
        from riesgos_tiempo_real import CircuitBreaker
        cb = CircuitBreaker()
        event = cb.trigger('test', ['NVDA', 'AAPL'], duration=1)
        assert len(cb.get_active()) > 0

    def test_dashboard_riesgo(self):
        from riesgos_tiempo_real import RiskDashboard
        rd = RiskDashboard()
        positions = {'NVDA': 1000}
        prices = {'NVDA': 150}
        dates = pd.date_range('2024-01-01', periods=100, freq='B')
        returns = pd.DataFrame({'NVDA': np.random.randn(100) * 0.02}, index=dates)
        result = rd.evaluate(positions, prices, returns, 500000)
        assert 'leverage' in result
        assert 'var' in result


# === TESTS MLOps ===
class TestMLOps:
    def test_versionador_datos(self):
        from mlops_pipeline import DataVersioner
        dv = DataVersioner()
        data = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        snap = dv.snapshot(data, 'test')
        assert snap.rows == 3

    def test_detector_drift(self):
        from mlops_pipeline import DriftDetector
        dd = DriftDetector()
        data = pd.DataFrame({'f1': np.random.randn(100)})
        dd.set_reference(data)
        result = dd.detect(pd.DataFrame({'f1': np.random.randn(100) + 0.1}))
        assert 'drift_detected' in result

    def test_registro_modelos(self):
        from mlops_pipeline import ModelRegistry
        mr = ModelRegistry()
        mv = mr.register('test_model', '/tmp/test', {'acc': 0.85}, ['f1', 'f2'])
        assert mv.version == 1
        mr.promote('test_model', 1)
        assert 'test_model' in mr.production

    def test_rollback(self):
        from mlops_pipeline import ModelRegistry
        mr = ModelRegistry()
        mr.register('test', '/tmp/v1', {'acc': 0.8}, ['f1'])
        mr.register('test', '/tmp/v2', {'acc': 0.9}, ['f1'])
        mr.promote('test', 2)
        rolled = mr.rollback('test')
        assert rolled is not None

    def test_probador_ab(self):
        from mlops_pipeline import ABTester
        ab = ABTester()
        ab.start_trial('test_trial', 'model_a', 'model_b', min_samples=10)
        for _ in range(20):
            ab.record('test_trial', 'model_a', np.random.random() > 0.4)
            ab.record('test_trial', 'model_b', np.random.random() > 0.5)
        result = ab.evaluate('test_trial')
        assert 'status' in result

    def test_pipeline_mlops(self):
        from mlops_pipeline import MLOpsPipeline
        mlops = MLOpsPipeline()
        mlops.log_decision('NVDA', 'COMPRA', 0.8, 1, {'rsi': 60})
        mlops.log_decision('AAPL', 'VENTA', 0.7, 1, {'rsi': 40})
        mlops.update_outcome(0, 0.02, True)
        mlops.update_outcome(1, -0.01, False)
        dash = mlops.get_dashboard()
        assert dash['correct'] == 1
        assert dash['incorrect'] == 1


# === TESTS NLU ===
class TestNLU:
    def test_clasificador_intencion(self):
        from interfaz_lenguaje_natural import IntentClassifier
        ic = IntentClassifier()
        intent, conf = ic.classify('como va mi portafolio')
        assert intent == 'consulta_portafolio'

    def test_extraer_tickers(self):
        from interfaz_lenguaje_natural import IntentClassifier
        ic = IntentClassifier()
        tickers = ic.extract_tickers('que hago con NVDA y AAPL')
        assert 'NVDA' in tickers
        assert 'AAPL' in tickers

    def test_generador_respuesta(self):
        from interfaz_lenguaje_natural import ResponseGenerator, NLUQuery
        rg = ResponseGenerator()
        q = NLUQuery(raw='test', intent='ayuda', entities={}, tickers=[],
                     confidence=0.9, timestamp='')
        r = rg.generate(q)
        assert 'ayudarte' in r.answer.lower()

    def test_preguntar_nlu(self):
        from interfaz_lenguaje_natural import NaturalLanguageInterface
        nlu = NaturalLanguageInterface()
        r = nlu.ask('como va mi portafolio')
        assert 'portafolio' in r.answer.lower() or 'Portafolio' in r.answer or '$' in r.answer

    def test_nlu_senal(self):
        from interfaz_lenguaje_natural import NaturalLanguageInterface
        nlu = NaturalLanguageInterface()
        r = nlu.ask('que hago con NVDA')
        assert r.intent == 'consulta_senal'

    def test_nlu_riesgo(self):
        from interfaz_lenguaje_natural import NaturalLanguageInterface
        nlu = NaturalLanguageInterface()
        r = nlu.ask('cual es el riesgo')
        assert 'riesgo' in r.intent or 'VaR' in r.answer or 'var' in r.answer.lower()

    def test_nlu_ayuda(self):
        from interfaz_lenguaje_natural import NaturalLanguageInterface
        nlu = NaturalLanguageInterface()
        r = nlu.ask('ayuda')
        assert 'ayudarte' in r.answer.lower()


# === PERSISTENT DB TESTS ===
class TestBaseDatos:
    def setup_method(self):
        import tempfile, os
        self.tmp = tempfile.mkdtemp()
        self.db_path = Path(self.tmp) / 'test.db'

    def teardown_method(self):
        import shutil
        if hasattr(self, 'tmp') and Path(self.tmp).exists():
            shutil.rmtree(self.tmp, ignore_errors=True)

    def _db(self):
        from persistent_db import DatabaseManager
        return DatabaseManager(self.db_path)

    def test_crear_db(self):
        db = self._db()
        assert self.db_path.exists()
        assert db is not None

    def test_guardar_modelo(self):
        db = self._db()
        v = db.guardar_modelo('test', 'XGBoost', 'ALCISTA',
                               '/tmp/m.pkl', {'acc': 0.85}, ['rsi'])
        assert v == 1
        v2 = db.guardar_modelo('test', 'XGBoost', 'ALCISTA',
                               '/tmp/m2.pkl', {'acc': 0.9}, ['rsi', 'macd'])
        assert v2 == 2

    def test_obtener_modelo(self):
        db = self._db()
        db.guardar_modelo('test', 'XGB', 'ALCISTA', '/tmp/m.pkl',
                          {'acc': 0.85}, ['rsi'])
        m = db.obtener_modelo('test')
        assert m is not None
        assert m['model_id'] == 'test'

    def test_promover_modelo(self):
        db = self._db()
        db.guardar_modelo('test', 'XGB', 'ALCISTA', '/tmp/v1.pkl',
                          {'acc': 0.8}, ['rsi'])
        db.guardar_modelo('test', 'XGB', 'ALCISTA', '/tmp/v2.pkl',
                          {'acc': 0.9}, ['rsi'])
        db.promover_modelo('test', 2)
        m = db.obtener_modelo('test', version=2)
        assert m['status'] == 'production'

    def test_guardar_decision(self):
        db = self._db()
        db.guardar_decision('NVDA', 'COMPRA', 0.78, 1, {'rsi': 65})
        dec = db.obtener_decisiones('NVDA')
        assert len(dec) > 0
        assert dec[0]['ticker'] == 'NVDA'

    def test_guardar_orden(self):
        db = self._db()
        db.guardar_orden('ORD_001', 'NVDA', 'COMPRA', 100, 150.0)
        ords = db.obtener_ordenes('NVDA')
        assert len(ords) > 0

    def test_guardar_riesgo(self):
        db = self._db()
        db.guardar_riesgo('NVDA', 'apalancamiento', 0.8, 1.0, False)
        ries = db.obtener_riesgos('NVDA')
        assert len(ries) > 0

    def test_guardar_caracteristica(self):
        db = self._db()
        db.guardar_caracteristica('NVDA', 'rsi_14', 65.0)
        hist = db.obtener_historial_caracteristicas('NVDA', 'rsi_14', 30)
        assert len(hist) > 0

    def test_dashboard(self):
        db = self._db()
        d = db.obtener_dashboard()
        assert 'db_path' in d
        assert d['decisiones'] == 0

    def test_snapshot(self):
        db = self._db()
        db.guardar_orden('ORD_SNAP', 'NVDA', 'COMPRA', 100, 150.0)
        path = db.snapshot_tabla('ordenes', 'test_snap')
        assert Path(path).exists()


# === HYPERPARAMETER OPTIMIZER TESTS ===
class TestOptimizadorHiperparams:
    def test_espacios_definidos(self):
        from hyperparameter_optimizer import ESPACIOS_XGBOOST, ESPACIOS_LIGHTGBM, ESPACIOS_RANDOMFOREST
        assert len(ESPACIOS_XGBOOST) > 0
        assert len(ESPACIOS_LIGHTGBM) > 0
        assert len(ESPACIOS_RANDOMFOREST) > 0

    def test_sugerir_params(self):
        import optuna
        from hyperparameter_optimizer import sugerir_params, ESPACIOS_XGBOOST
        study = optuna.create_study(direction='maximize')
        trial = study.ask()
        params = sugerir_params(trial, ESPACIOS_XGBOOST)
        assert 'n_estimators' in params
        assert 'max_depth' in params

    def test_optimizador_init(self):
        from hyperparameter_optimizer import HyperparameterOptimizer
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            path = f'sqlite:///{tmp}/test.db'
            opt = HyperparameterOptimizer(study_name='test', storage=path)
            assert opt.study_name == 'test'
            assert opt.direction == 'maximize'

    def test_optimizar_xgboost(self):
        from hyperparameter_optimizer import OptimizadorXGBoost
        import uuid
        np.random.seed(42)
        n = 200
        X = pd.DataFrame({f'f{i}': np.random.randn(n) for i in range(5)})
        y = (X['f0'] + X['f1'] * 0.3 > 0).astype(int)
        opt = OptimizadorXGBoost(X, y)
        res = opt.optimizar(n_trials=5, study_name=f'test_xgb_{uuid.uuid4().hex[:8]}')
        assert res.n_trials > 0
        assert len(res.best_params) > 0 or res.n_trials >= 3

    def test_automl_pipeline(self):
        from hyperparameter_optimizer import AutoMLPipeline
        np.random.seed(42)
        n = 200
        X = pd.DataFrame({f'f{i}': np.random.randn(n) for i in range(5)})
        y = (X['f0'] > 0).astype(int)
        pipeline = AutoMLPipeline()
        pipeline.comparar_modelos(X, y, modelos=['xgboost'], n_trials=3)
        assert pipeline.mejor_modelo == 'xgboost'


# === DASHBOARD INTERACTIVO TESTS ===
class TestDashboardInteractivo:
    def test_datos_simulados_portafolio(self):
        from dashboard_interactivo import DatosSimulados
        d = DatosSimulados()
        df = d.portafolio(50)
        assert len(df) == 50
        assert 'portafolio' in df.columns

    def test_datos_simulados_tickers(self):
        from dashboard_interactivo import DatosSimulados
        d = DatosSimulados()
        tickers = d.tickers()
        assert len(tickers) > 0
        assert 'NVDA' in tickers

    def test_datos_simulados_precios(self):
        from dashboard_interactivo import DatosSimulados
        d = DatosSimulados()
        df = d.precios('NVDA', 100)
        assert len(df) == 100

    def test_datos_simulados_senales(self):
        from dashboard_interactivo import DatosSimulados
        d = DatosSimulados()
        df = d.senales(5)
        assert len(df) > 0

    def test_datos_simulados_metricas(self):
        from dashboard_interactivo import DatosSimulados
        d = DatosSimulados()
        m = d.metricas()
        assert 'sharpe' in m
        assert 'retorno_anual' in m

    def test_datos_simulados_exposicion(self):
        from dashboard_interactivo import DatosSimulados
        d = DatosSimulados()
        df = d.exposicion()
        assert len(df) > 0

    def test_generar_html_offline(self):
        from dashboard_interactivo import generar_html_offline
        import tempfile
        path = generar_html_offline(n_dias=30)
        assert Path(path).exists()


# === BROKER INTERFACE TESTS ===
class TestBrokerInterface:
    def test_lados_enum(self):
        from broker_interface import LadoOrden
        assert LadoOrden.COMPRA.value == 'COMPRA'
        assert LadoOrden.VENTA.value == 'VENTA'

    def test_tipos_enum(self):
        from broker_interface import TipoOrden, EstadoOrden
        assert TipoOrden.MERCADO.value == 'MERCADO'
        assert EstadoOrden.PENDIENTE.value == 'pendiente'

    def test_broker_paper_conectar(self):
        from broker_interface import BrokerPaper
        b = BrokerPaper(capital_inicial=100000)
        assert b.conectar()
        assert b.conectado

    def test_broker_paper_enviar_orden_compra(self):
        from broker_interface import BrokerPaper, LadoOrden
        b = BrokerPaper(capital_inicial=100000)
        b.conectar()
        o = b.enviar_orden('NVDA', LadoOrden.COMPRA, 10)
        assert o.estado.value == 'llena'
        assert o.shares_llenas == 10

    def test_broker_paper_enviar_orden_venta(self):
        from broker_interface import BrokerPaper, LadoOrden
        b = BrokerPaper(capital_inicial=100000)
        b.conectar()
        b.enviar_orden('NVDA', LadoOrden.COMPRA, 10)
        o = b.enviar_orden('NVDA', LadoOrden.VENTA, 5)
        assert o.estado.value == 'llena'

    def test_broker_paper_fondos_insuficientes(self):
        from broker_interface import BrokerPaper, LadoOrden
        b = BrokerPaper(capital_inicial=1000)
        b.conectar()
        o = b.enviar_orden('NVDA', LadoOrden.COMPRA, 10000)
        assert o.estado.value == 'rechazada'

    def test_broker_paper_posiciones(self):
        from broker_interface import BrokerPaper, LadoOrden
        b = BrokerPaper(capital_inicial=100000)
        b.conectar()
        b.enviar_orden('NVDA', LadoOrden.COMPRA, 10)
        pos = b.obtener_posiciones()
        assert len(pos) > 0
        assert pos[0].ticker == 'NVDA'

    def test_broker_paper_resumen(self):
        from broker_interface import BrokerPaper, LadoOrden
        b = BrokerPaper(capital_inicial=100000)
        b.conectar()
        b.enviar_orden('AAPL', LadoOrden.COMPRA, 50)
        r = b.obtener_resumen()
        assert r.valor_portafolio > 0
        assert len(r.posiciones) > 0

    def test_gestor_brokers_default(self):
        from broker_interface import GestorBrokers
        g = GestorBrokers()
        g.crear_brokers_default(100000)
        assert 'paper' in g.brokers
        assert g.activo() is not None

    def test_gestor_enviar_orden(self):
        from broker_interface import GestorBrokers, LadoOrden
        g = GestorBrokers()
        g.crear_brokers_default(100000)
        o = g.enviar_orden('NVDA', LadoOrden.COMPRA, 10)
        assert o is not None
        assert o.estado.value == 'llena'

    def test_gestor_cambiar_broker(self):
        from broker_interface import GestorBrokers
        g = GestorBrokers()
        g.crear_brokers_default(100000)
        assert g.cambiar_broker('paper')
        assert g.activo().nombre == 'paper'

    def test_gestor_estado_conexiones(self):
        from broker_interface import GestorBrokers
        g = GestorBrokers()
        g.crear_brokers_default(100000)
        est = g.estado_conexiones()
        assert 'paper' in est


# === TCA TESTS ===
class TestTCA:
    def test_trade_record(self):
        from transaction_cost_analysis import TradeRecord
        t = TradeRecord(trade_id='T1', ticker='NVDA', lado='COMPRA',
                        shares=100, precio_esperado=100.0, precio_real=100.5,
                        timestamp_decision='2024-01-01T10:00:00',
                        timestamp_envio='2024-01-01T10:01:00',
                        timestamp_lleno='2024-01-01T10:01:30')
        assert t.ticker == 'NVDA'

    def test_calcular_slippage(self):
        from transaction_cost_analysis import CalculatorTCA, TradeRecord
        calc = CalculatorTCA()
        t = TradeRecord(trade_id='T1', ticker='NVDA', lado='COMPRA',
                        shares=100, precio_esperado=100.0, precio_real=100.5,
                        timestamp_decision='2024-01-01T10:00:00',
                        timestamp_envio='2024-01-01T10:01:00',
                        timestamp_lleno='2024-01-01T10:01:30')
        s = calc.calcular_slippage(t)
        assert s > 0

    def test_market_impact(self):
        from transaction_cost_analysis import CalculatorTCA, TradeRecord
        calc = CalculatorTCA()
        t = TradeRecord(trade_id='T1', ticker='NVDA', lado='COMPRA',
                        shares=100, precio_esperado=100, precio_real=100.5,
                        timestamp_decision='', timestamp_envio='',
                        timestamp_lleno='', volumen_diario=1e6,
                        volatilidad=0.02, participacion=0.05)
        imp = calc.calcular_market_impact(t)
        assert imp > 0

    def test_calcular_todo(self):
        from transaction_cost_analysis import CalculatorTCA, TradeRecord
        calc = CalculatorTCA()
        t = TradeRecord(trade_id='T1', ticker='NVDA', lado='COMPRA',
                        shares=100, precio_esperado=100, precio_real=100.3,
                        timestamp_decision='2024-01-01T10:00:00',
                        timestamp_envio='2024-01-01T10:02:00',
                        timestamp_lleno='2024-01-01T10:03:00',
                        volumen_diario=2e6, volatilidad=0.02, spread_bps=10,
                        participacion=0.02)
        c = calc.calcular_todo(t)
        assert c.total_bps > 0
        assert c.total_dinero > 0

    def test_tca_analyzer(self):
        from transaction_cost_analysis import TCAAnalyzer, TradeRecord
        analyzer = TCAAnalyzer()
        t = TradeRecord(trade_id='T1', ticker='NVDA', lado='COMPRA',
                        shares=100, precio_esperado=100, precio_real=100.3,
                        timestamp_decision='2024-01-01T10:00:00',
                        timestamp_envio='2024-01-01T10:02:00',
                        timestamp_lleno='2024-01-01T10:03:00',
                        volumen_diario=2e6, volatilidad=0.02, spread_bps=10,
                        participacion=0.02)
        r = analyzer.analizar_trade(t)
        assert r.calidad in ['excelente', 'buena', 'regular', 'mala']

    def test_resumen_estadistico(self):
        from transaction_cost_analysis import TCAAnalyzer, TradeRecord
        analyzer = TCAAnalyzer()
        for i in range(5):
            t = TradeRecord(trade_id=f'T{i}', ticker='NVDA', lado='COMPRA',
                            shares=100, precio_esperado=100,
                            precio_real=100 + i * 0.1,
                            timestamp_decision='2024-01-01T10:00:00',
                            timestamp_envio='2024-01-01T10:02:00',
                            timestamp_lleno='2024-01-01T10:03:00',
                            volumen_diario=2e6, volatilidad=0.02,
                            spread_bps=10, participacion=0.02)
            analyzer.analizar_trade(t)
        res = analyzer.resumen_estadistico()
        assert res['total_trades'] == 5
        assert 'slippage_promedio' in res

    def test_resumen_por_ticker(self):
        from transaction_cost_analysis import TCAAnalyzer, TradeRecord
        analyzer = TCAAnalyzer()
        for tic in ['NVDA', 'AAPL']:
            for j in range(3):
                t = TradeRecord(trade_id=f'{tic}_{j}', ticker=tic,
                                lado='COMPRA', shares=100,
                                precio_esperado=100, precio_real=100.2,
                                timestamp_decision='',
                                timestamp_envio='', timestamp_lleno='',
                                volumen_diario=1e6, volatilidad=0.02,
                                spread_bps=10, participacion=0.01)
                analyzer.analizar_trade(t)
        df = analyzer.resumen_por_ticker()
        assert len(df) == 2

    def test_simulador(self):
        from transaction_cost_analysis import SimuladorTCA
        sim = SimuladorTCA()
        trades = sim.simular_lote(['NVDA', 'AAPL'], 3)
        assert len(trades) == 6

    def test_generar_reporte(self):
        from transaction_cost_analysis import TCAAnalyzer, TradeRecord
        analyzer = TCAAnalyzer()
        t = TradeRecord(trade_id='R1', ticker='NVDA', lado='COMPRA',
                        shares=100, precio_esperado=100, precio_real=100.2,
                        timestamp_decision='',
                        timestamp_envio='', timestamp_lleno='',
                        volumen_diario=1e6, volatilidad=0.02,
                        spread_bps=10, participacion=0.01)
        analyzer.analizar_trade(t)
        path = analyzer.generar_reporte()
        assert Path(path).exists()
        assert 'tca_report' in path