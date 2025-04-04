import os
from dotenv import load_dotenv

class ConfigManager:
    """設定管理クラス"""
    
    def __init__(self):
        """コンフィグマネージャの初期化"""
        # 環境変数の読み込み
        load_dotenv()
        
        # API設定
        self.api_key = os.getenv("BINANCE_API_KEY")
        self.api_secret = os.getenv("BINANCE_API_SECRET")
        
        # 取引設定
        self.symbol = os.getenv("SYMBOL", "BTCUSDT")
        self.interval = os.getenv("INTERVAL", "1h")
        self.trade_quantity = float(os.getenv("QUANTITY", "0.001"))
        
        # 基本的な戦略パラメータ
        self.short_window = int(os.getenv("SHORT_WINDOW", "9"))
        self.long_window = int(os.getenv("LONG_WINDOW", "21"))
        self.stop_loss_percent = float(os.getenv("STOP_LOSS_PERCENT", "2.0"))
        self.take_profit_percent = float(os.getenv("TAKE_PROFIT_PERCENT", "5.0"))
        
        # RSI設定
        self.rsi_period = int(os.getenv("RSI_PERIOD", "14"))
        self.rsi_oversold = float(os.getenv("RSI_OVERSOLD", "30"))
        self.rsi_overbought = float(os.getenv("RSI_OVERBOUGHT", "70"))
        
        # MACD設定
        self.macd_fast = int(os.getenv("MACD_FAST", "12"))
        self.macd_slow = int(os.getenv("MACD_SLOW", "26"))
        self.macd_signal = int(os.getenv("MACD_SIGNAL", "9"))
        
        # ボリンジャーバンド設定
        self.bb_period = int(os.getenv("BB_PERIOD", "20"))
        self.bb_std = float(os.getenv("BB_STD", "2.0"))
        
        # 複合シグナルの重み付け設定
        self.weight_ma = float(os.getenv("WEIGHT_MA", "0.3"))
        self.weight_rsi = float(os.getenv("WEIGHT_RSI", "0.2"))
        self.weight_macd = float(os.getenv("WEIGHT_MACD", "0.2"))
        self.weight_bb = float(os.getenv("WEIGHT_BB", "0.2"))
        self.weight_breakout = float(os.getenv("WEIGHT_BREAKOUT", "0.1"))
        
        # シグナル閾値
        self.buy_threshold = float(os.getenv("BUY_THRESHOLD", "0.5"))
        self.sell_threshold = float(os.getenv("SELL_THRESHOLD", "-0.5"))
        
        # フィルター設定
        self.use_complex_signal = os.getenv("USE_COMPLEX_SIGNAL", "true").lower() == "true"
        self.use_price_simulation = os.getenv("USE_PRICE_SIMULATION", "true").lower() == "true"
        
        # コストとスリッページ設定
        self.maker_fee = float(os.getenv("MAKER_FEE", "0.0010"))
        self.taker_fee = float(os.getenv("TAKER_FEE", "0.0010"))
        self.slippage_mean = float(os.getenv("SLIPPAGE_MEAN", "0.0005"))
        self.slippage_std = float(os.getenv("SLIPPAGE_STD", "0.0003"))
        
        # バックテスト設定
        self.backtest_days = int(os.getenv("BACKTEST_DAYS", "30"))
        self.execution_delay = int(os.getenv("EXECUTION_DELAY", "1"))
        self.price_simulation_steps = int(os.getenv("PRICE_SIMULATION_STEPS", "100"))
        self.use_cached_data = os.getenv("USE_CACHED_DATA", "true").lower() == "true"
        
        # リスク管理設定
        self.max_consecutive_losses = int(os.getenv("MAX_CONSECUTIVE_LOSSES", "3"))
        self.max_drawdown_limit = float(os.getenv("MAX_DRAWDOWN_LIMIT", "5.0"))
        self.risk_per_trade = float(os.getenv("RISK_PER_TRADE", "0.01"))
        
        # 最適化設定
        self.optimization_workers = int(os.getenv("OPTIMIZATION_WORKERS", "0"))  # 0=自動
        
        # 必要なディレクトリの確認
        self._ensure_directories()
    
    def _ensure_directories(self):
        """必要なディレクトリを作成"""
        for directory in ['data', 'logs', 'results', 'cache', 'models']:
            os.makedirs(directory, exist_ok=True)
    
    def to_dict(self):
        """設定を辞書として取得"""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
    
    def update(self, **kwargs):
        """
        設定を更新（型変換付き）
        
        Parameters:
        -----------
        **kwargs : 
            更新するパラメータと値
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                # 整数型パラメータは適切に変換
                if key in ["short_window", "long_window", "rsi_period", "macd_fast", 
                           "macd_slow", "macd_signal", "bb_period", "execution_delay", 
                           "price_simulation_steps", "max_consecutive_losses"]:
                    try:
                        value = int(value)
                    except (ValueError, TypeError):
                        # 変換できない場合は警告を出して元の値を使用
                        import logging
                        logging.warning(f"パラメータ '{key}' の値 '{value}' を整数に変換できません。デフォルト値を使用します。")
                        continue
                
                # 浮動小数点型パラメータの変換
                elif key in ["stop_loss_percent", "take_profit_percent", "rsi_oversold", 
                             "rsi_overbought", "bb_std", "weight_ma", "weight_rsi", 
                             "weight_macd", "weight_bb", "weight_breakout", "buy_threshold", 
                             "sell_threshold", "maker_fee", "taker_fee", "slippage_mean", 
                             "slippage_std", "max_drawdown_limit", "risk_per_trade"]:
                    try:
                        value = float(value)
                    except (ValueError, TypeError):
                        # 変換できない場合は警告を出して元の値を使用
                        import logging
                        logging.warning(f"パラメータ '{key}' の値 '{value}' を浮動小数点に変換できません。デフォルト値を使用します。")
                        continue
                
                # 設定を更新
                setattr(self, key, value)