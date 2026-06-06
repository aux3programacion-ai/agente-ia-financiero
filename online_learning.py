import json, os, sys, time
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

TICKERS_CORE = ['NVDA','MU','DELL','AVGO','DDOG','SMCI','SNOW','CRWD','NOW','TSM','ARM','OKTA','HPE','NTAP','CLS','AAPL','AMZN','GOOGL','META','MSFT','LLY','AMAT','LRCX','PANW','ORCL','HON','UBER','GE','COST','NEE']
DATA_DIR = os.environ.get('GITHUB_WORKSPACE', '.')
OUTPUT = os.path.join(DATA_DIR, 'Datos', 'online_learning.json')
MODEL_DIR = os.path.join(DATA_DIR, 'Datos', 'online_models')
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, 'Datos'), exist_ok=True)

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def extract_features(df):
    close = df['Close']
    features = pd.DataFrame(index=df.index)
    features['return_1d'] = close.pct_change()
    features['return_5d'] = close.pct_change(5)
    features['return_20d'] = close.pct_change(20)
    features['rsi'] = compute_rsi(close)
    features['vol_ratio'] = df['Volume'] / df['Volume'].rolling(50).mean()
    features['volatility'] = close.pct_change().rolling(20).std()
    features['sma50_dist'] = (close - close.rolling(50).mean()) / close.rolling(50).mean()
    features = features.dropna()
    return features

def main():
    print('[Online Learning] Entrenamiento incremental con SGD...')
    all_results = {}
    scaler = StandardScaler()
    fitted = False
    
    for ticker in TICKERS_CORE[:5]:
        try:
            print(f'  {ticker}...', end=' ')
            df = yf.download(ticker, period='1y', interval='1d', progress=False, auto_adjust=True)
            if df is None or df.empty:
                print('[NO DATA]')
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df = df.xs(ticker, axis=1, level=0) if ticker in df.columns.get_level_values(0) else df
            df = df.dropna(subset=['Close'])
            
            feats = extract_features(df)
            if len(feats) < 100:
                print('[SHORT]')
                continue
            
            # Target: next 5d direction
            target = (df['Close'].shift(-5) > df['Close']).astype(int)
            feats = feats.join(target).dropna()
            
            if len(feats) < 50:
                print('[SHORT]')
                continue
            
            X_all = feats.drop(columns=['target']).values
            y_all = feats['target'].values
            
            if not fitted:
                X_scaled = scaler.fit_transform(X_all)
                fitted = True
            else:
                X_scaled = scaler.transform(X_all)
            
            # Online learning: partial_fit in batches
            model_path = os.path.join(MODEL_DIR, f'{ticker}_sgd.pkl')
            model_file = f'{ticker}_sgd.pkl'
            model_full_path = os.path.join(MODEL_DIR, model_file)
            
            classes = np.array([0, 1])
            if os.path.exists(model_full_path):
                import pickle
                model = pickle.load(open(model_full_path, 'rb'))
                # Update with new data
                model.partial_fit(X_scaled, y_all, classes=classes)
            else:
                model = SGDClassifier(loss='log_loss', penalty='elasticnet', alpha=0.0001,
                                      learning_rate='adaptive', eta0=0.01, random_state=42)
                model.partial_fit(X_scaled, y_all, classes=classes)
            
            import pickle
            pickle.dump(model, open(model_full_path, 'wb'))
            
            # Evaluate
            proba = model.predict_proba(X_scaled[-1:])[0][1] * 100
            train_acc = model.score(X_scaled, y_all)
            
            # Rolling accuracy on last 30
            n_train = len(X_scaled)
            if n_train > 60:
                roll_preds = model.predict(X_scaled[-30:])
                roll_acc = np.mean(roll_preds == y_all[-30:])
            else:
                roll_acc = train_acc
            
            all_results[ticker] = {
                'prob_up_20d': round(proba, 1),
                'train_accuracy': round(train_acc, 4),
                'rolling_30d_accuracy': round(roll_acc, 4),
                'n_samples': len(feats),
                'model_version': time.strftime('%Y%m%d')
            }
            print(f'prob={proba:.0f}% acc={train_acc:.2f}')
        except Exception as e:
            print(f'[!] {e}')
    
    output = {
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'tickers': all_results
    }
    with open(OUTPUT, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2)
    print(f'\n[OK] {len(all_results)} tickers actualizados via online learning')

if __name__ == '__main__':
    main()
