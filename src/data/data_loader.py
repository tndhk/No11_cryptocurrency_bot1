import os
import time
import pickle
import traceback
import pandas as pd
from binance.client import Client
from binance.exceptions import BinanceAPIException
from src.utils.logger import default_logger as logger

class DataLoader:
    """
    取引所からのデータ取得とキャッシュを管理するクラス
    """
    
    def __init__(self, config):
        """
        初期化
        
        Parameters:
        -----------
        config: ConfigManager
            設定マネージャのインスタンス
        """
        self.config = config
        self.client = None
        
        # API制限処理のための変数
        self.api_request_count = 0
        self.last_api_reset = time.time()
        self.max_requests_per_minute = 1200  # Binanceの制限
    
    def _initialize_client(self):
        """APIクライアントを初期化（ライブモード用）"""
        if self.client is None:
            try:
                self.client = Client(self.config.api_key, self.config.api_secret)
                logger.info("Binance APIクライアント初期化成功")
            except Exception as e:
                logger.error(f"Binance APIクライアント初期化失敗: {e}")
                raise
    
    def _check_api_rate_limit(self):
        """API呼び出し制限をチェックし、必要に応じて待機"""
        current_time = time.time()
        # 1分経過していたらカウンターをリセット
        if current_time - self.last_api_reset > 60:
            self.api_request_count = 0
            self.last_api_reset = current_time
        
        # 制限に近づいたら待機
        if self.api_request_count > self.max_requests_per_minute * 0.9:
            wait_time = 60 - (current_time - self.last_api_reset)
            if wait_time > 0:
                logger.warning(f"API制限に近づいています。{wait_time:.2f}秒待機します")
                time.sleep(wait_time)
                self.api_request_count = 0
                self.last_api_reset = time.time()
        
        self.api_request_count += 1
    
    def get_historical_data(self, start_time=None, end_time=None, symbol=None, 
                            interval=None, limit=1000, is_backtest=False):
        """
        バックテスト用かライブ用かに関わらず一貫した方法で
        歴史データを取得する汎用メソッド
        
        Parameters:
        -----------
        start_time : int, optional
            取得開始時間（ミリ秒）
        end_time : int, optional
            取得終了時間（ミリ秒）
        symbol : str, optional
            取引ペア（指定がなければ設定値）
        interval : str, optional
            時間間隔（指定がなければ設定値）
        limit : int
            取得するロウソク足の数
        is_backtest : bool
            バックテストモードかどうか
            
        Returns:
        --------
        pandas.DataFrame
            OHLCV データ
        """
        # シンボルとインターバルの設定
        symbol = symbol or self.config.symbol
        interval = interval or self.config.interval
        
        # キャッシュファイルパスの設定
        cache_file = f"cache/{symbol}_{interval}_history.pkl"
        
        # バックテストモードでキャッシュを使用する場合
        if is_backtest and self.config.use_cached_data and os.path.exists(cache_file):
            try:
                # キャッシュからデータを読み込み
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                
                # キャッシュの有効性確認（最新のデータが含まれているか）
                if end_time and cached_data['timestamp'].max() >= pd.to_datetime(end_time, unit='ms'):
                    logger.info(f"キャッシュからデータを読み込みました: {len(cached_data)} ロウソク足")
                    
                    # 指定された期間のデータを抽出
                    if start_time:
                        start_dt = pd.to_datetime(start_time, unit='ms')
                        filtered_data = cached_data[cached_data['timestamp'] >= start_dt]
                    else:
                        filtered_data = cached_data
                    
                    if end_time:
                        end_dt = pd.to_datetime(end_time, unit='ms')
                        filtered_data = filtered_data[filtered_data['timestamp'] <= end_dt]
                    
                    return filtered_data
                else:
                    logger.info("キャッシュが古いため、新しいデータを取得します")
            except Exception as e:
                logger.warning(f"キャッシュ読み込みエラー: {e}")
        
        # APIからデータを取得
        try:
            # ライブ環境でのみAPIクライアントを初期化
            self._initialize_client()
            self._check_api_rate_limit()
            
            logger.info(f"APIからデータを取得: {symbol}, {interval}")
            
            # start_timeとend_timeが指定されている場合
            if start_time and end_time:
                klines = self.client.get_historical_klines(
                    symbol=symbol,
                    interval=interval,
                    start_str=str(start_time),
                    end_str=str(end_time),
                    limit=limit
                )
            # limit指定のみの場合
            else:
                klines = self.client.get_klines(
                    symbol=symbol,
                    interval=interval,
                    limit=limit
                )
            
            # データフレームに変換
            data = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            # データ型変換
            data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
            for col in ['open', 'high', 'low', 'close', 'volume']:
                data[col] = data[col].astype(float)
            
            # バックテスト用にデータをキャッシュ
            if is_backtest and len(data) > 0:
                self._cache_data(data, cache_file)
            
            return data
                
        except BinanceAPIException as e:
            logger.error(f"APIエラー: {e}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"データ取得エラー: {e}")
            logger.error(traceback.format_exc())
            return pd.DataFrame()
    
    def _cache_data(self, data, cache_file):
        """データをキャッシュに保存"""
        try:
            # 既存のキャッシュと結合
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, 'rb') as f:
                        cached_data = pickle.load(f)
                    
                    # 重複を避けるために結合
                    combined_data = pd.concat([cached_data, data])
                    combined_data = combined_data.drop_duplicates(subset=['timestamp'])
                    
                    # 時間でソート
                    combined_data = combined_data.sort_values('timestamp')
                    
                    with open(cache_file, 'wb') as f:
                        pickle.dump(combined_data, f)
                        
                    logger.info(f"キャッシュ更新: {len(combined_data)} ロウソク足")
                except Exception as e:
                    logger.warning(f"キャッシュ更新エラー: {e}")
                    # 新規キャッシュを作成
                    with open(cache_file, 'wb') as f:
                        pickle.dump(data, f)
            else:
                # 新規キャッシュを作成
                with open(cache_file, 'wb') as f:
                    pickle.dump(data, f)
                logger.info(f"新規キャッシュ作成: {len(data)} ロウソク足")
        except Exception as e:
            logger.error(f"キャッシュ保存エラー: {e}")