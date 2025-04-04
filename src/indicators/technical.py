import pandas as pd
import numpy as np
from src.utils.logger import default_logger as logger

class IndicatorCalculator:
    """
    テクニカル指標計算クラス
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
    
    def calculate_indicators(self, data):
        """
        テクニカル指標を計算
        
        Parameters:
        -----------
        data : pandas.DataFrame
            OHLCV データ
            
        Returns:
        --------
        pandas.DataFrame
            指標が追加されたデータフレーム
        """
        if data.empty:
            return data
            
        df = data.copy()
        
        # 移動平均の計算
        self._calculate_moving_averages(df)
        
        # RSIの計算
        self._calculate_rsi(df)
        
        # ボリンジャーバンドの計算
        self._calculate_bollinger_bands(df)
        
        # MACDの計算
        self._calculate_macd(df)
        
        # ブレイクアウトの計算
        self._calculate_breakout(df)
        
        # 出来高指標の計算
        self._calculate_volume_indicators(df)
        
        # ATRの計算
        self._calculate_atr(df)
        
        # 複合シグナルの計算
        self._calculate_complex_signal(df)
        
        return df
    
    def _calculate_moving_averages(self, df):
        """移動平均を計算"""
        # パラメータのサニタイズ - 浮動小数点から整数値への変換
        try:
            short_window = max(1, int(self.config.short_window))  # 最小値1、整数化
            long_window = max(2, int(self.config.long_window))    # 最小値2、整数化
        except (ValueError, TypeError):
            logger.warning(f"移動平均のパラメータが無効です: short_window={self.config.short_window}, long_window={self.config.long_window}")
            # デフォルト値に設定
            short_window = 9
            long_window = 21
            logger.info(f"デフォルト値を使用: short_window={short_window}, long_window={long_window}")
        
        # データが足りない場合のチェック
        if len(df) < long_window + 1:
            logger.warning(f"データ不足: 長期移動平均（{long_window}）を計算するには少なくとも{long_window+1}データポイントが必要です")
            # 最小限のデータポイント数で調整
            short_window = min(short_window, max(1, len(df) // 3))
            long_window = min(long_window, max(2, len(df) // 2))
            logger.info(f"パラメータを調整: short_window={short_window}, long_window={long_window}")
        
        # 単純移動平均（SMA）
        df['SMA_short'] = df['close'].rolling(window=short_window).mean()
        df['SMA_long'] = df['close'].rolling(window=long_window).mean()
        
        try:
            # 指数移動平均（EMA）
            df['EMA_short'] = df['close'].ewm(span=short_window, adjust=False).mean()
            df['EMA_long'] = df['close'].ewm(span=long_window, adjust=False).mean()
            
            # 移動平均クロスオーバーシグナルの計算
            df['ma_signal'] = 0
            df.loc[df['EMA_short'] > df['EMA_long'], 'ma_signal'] = 1
            df.loc[df['EMA_short'] < df['EMA_long'], 'ma_signal'] = -1
        except Exception as e:
            logger.error(f"EMA計算エラー: {e}")
            # フォールバック: SMAを使用
            df['EMA_short'] = df['SMA_short']
            df['EMA_long'] = df['SMA_long']
            
            # SMAベースのシグナル
            df['ma_signal'] = 0
            df.loc[df['SMA_short'] > df['SMA_long'], 'ma_signal'] = 1
            df.loc[df['SMA_short'] < df['SMA_long'], 'ma_signal'] = -1
    
    def _calculate_rsi(self, df):
        """RSIを計算"""
        try:
            # パラメータのサニタイズ
            rsi_period = max(2, int(self.config.rsi_period))
            
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            avg_gain = gain.rolling(window=rsi_period).mean()
            avg_loss = loss.rolling(window=rsi_period).mean()
            
            rs = avg_gain / avg_loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # RSIベースのシグナル
            df['rsi_signal'] = 0
            df.loc[df['RSI'] < self.config.rsi_oversold, 'rsi_signal'] = 1  # 買い
            df.loc[df['RSI'] > self.config.rsi_overbought, 'rsi_signal'] = -1  # 売り
        except Exception as e:
            logger.error(f"RSI計算エラー: {e}")
            # フォールバック値を設定
            df['RSI'] = 50  # 中立的な値
            df['rsi_signal'] = 0
    
    def _calculate_bollinger_bands(self, df):
        """ボリンジャーバンドを計算"""
        try:
            # パラメータのサニタイズ
            bb_period = max(2, int(self.config.bb_period))
            
            df['BB_middle'] = df['close'].rolling(window=bb_period).mean()
            df['BB_std'] = df['close'].rolling(window=bb_period).std()
            df['BB_upper'] = df['BB_middle'] + self.config.bb_std * df['BB_std']
            df['BB_lower'] = df['BB_middle'] - self.config.bb_std * df['BB_std']
            
            # ボリンジャーバンドベースのシグナル
            df['bb_signal'] = 0
            df.loc[df['close'] < df['BB_lower'], 'bb_signal'] = 1  # 買い（下限突破）
            df.loc[df['close'] > df['BB_upper'], 'bb_signal'] = -1  # 売り（上限突破）
        except Exception as e:
            logger.error(f"ボリンジャーバンド計算エラー: {e}")
            # フォールバック値
            df['BB_middle'] = df['close']
            df['BB_upper'] = df['close'] * 1.02
            df['BB_lower'] = df['close'] * 0.98
            df['bb_signal'] = 0
    
    def _calculate_macd(self, df):
        """MACDを計算"""
        try:
            # パラメータのサニタイズ
            macd_fast = max(2, int(self.config.macd_fast))
            macd_slow = max(macd_fast + 1, int(self.config.macd_slow))
            macd_signal = max(2, int(self.config.macd_signal))
            
            df['EMA_' + str(macd_fast)] = df['close'].ewm(span=macd_fast, adjust=False).mean()
            df['EMA_' + str(macd_slow)] = df['close'].ewm(span=macd_slow, adjust=False).mean()
            df['MACD'] = df['EMA_' + str(macd_fast)] - df['EMA_' + str(macd_slow)]
            df['MACD_signal'] = df['MACD'].ewm(span=macd_signal, adjust=False).mean()
            df['MACD_hist'] = df['MACD'] - df['MACD_signal']
            
            # MACDベースのシグナル
            df['macd_signal'] = 0
            df.loc[(df['MACD'] > df['MACD_signal']) & (df['MACD'].shift(1) <= df['MACD_signal'].shift(1)), 'macd_signal'] = 1  # 買い（MACD上抜け）
            df.loc[(df['MACD'] < df['MACD_signal']) & (df['MACD'].shift(1) >= df['MACD_signal'].shift(1)), 'macd_signal'] = -1  # 売り（MACD下抜け）
        except Exception as e:
            logger.error(f"MACD計算エラー: {e}")
            # フォールバック値
            df['MACD'] = 0
            df['MACD_signal'] = 0
            df['MACD_hist'] = 0
            df['macd_signal'] = 0
    
    def _calculate_breakout(self, df):
        """ブレイクアウトを計算"""
        try:
            n_periods = 14
            df['highest_high'] = df['high'].rolling(window=n_periods).max()
            df['lowest_low'] = df['low'].rolling(window=n_periods).min()
            
            # ブレイクアウトベースのシグナル
            df['breakout_signal'] = 0
            df.loc[df['close'] > df['highest_high'].shift(1), 'breakout_signal'] = 1  # 買い（高値ブレイク）
            df.loc[df['close'] < df['lowest_low'].shift(1), 'breakout_signal'] = -1  # 売り（安値ブレイク）
        except Exception as e:
            logger.error(f"ブレイクアウト計算エラー: {e}")
            # フォールバック値
            df['highest_high'] = df['high']
            df['lowest_low'] = df['low']
            df['breakout_signal'] = 0
    
    def _calculate_volume_indicators(self, df):
        """出来高指標を計算"""
        try:
            df['volume_change'] = df['volume'].pct_change()
            df['volume_ma'] = df['volume'].rolling(window=20).mean()
            
            # 出来高ベースのシグナル
            df['volume_signal'] = 0
            df.loc[(df['volume'] > df['volume_ma'] * 1.5) & (df['close'] > df['open']), 'volume_signal'] = 1  # 大出来高の上昇
            df.loc[(df['volume'] > df['volume_ma'] * 1.5) & (df['close'] < df['open']), 'volume_signal'] = -1  # 大出来高の下落
        except Exception as e:
            logger.error(f"出来高指標計算エラー: {e}")
            # フォールバック値
            df['volume_change'] = 0
            df['volume_ma'] = df['volume']
            df['volume_signal'] = 0
    
    def _calculate_atr(self, df):
        """ATR（Average True Range）を計算"""
        try:
            high_low = df['high'] - df['low']
            high_close = abs(df['high'] - df['close'].shift())
            low_close = abs(df['low'] - df['close'].shift())
            
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            df['ATR'] = tr.rolling(14).mean()
        except Exception as e:
            logger.error(f"ATR計算エラー: {e}")
            # フォールバック値
            df['ATR'] = (df['high'] - df['low']) / 2  # 単純な代替指標
    
    def _calculate_complex_signal(self, df):
        """複合シグナルを計算"""
        try:
            if self.config.use_complex_signal:
                # すべての必要な列が存在するか確認
                required_columns = ['ma_signal', 'rsi_signal', 'macd_signal', 'bb_signal', 'breakout_signal']
                for col in required_columns:
                    if col not in df.columns:
                        df[col] = 0  # 不足している列を0で初期化
                
                df['complex_signal'] = (
                    self.config.weight_ma * df['ma_signal'] +
                    self.config.weight_rsi * df['rsi_signal'] +
                    self.config.weight_macd * df['macd_signal'] +
                    self.config.weight_bb * df['bb_signal'] +
                    self.config.weight_breakout * df['breakout_signal']
                )
                
                # シグナルのバイナリ化（閾値に基づく）
                df['signal'] = 0
                df.loc[df['complex_signal'] >= self.config.buy_threshold, 'signal'] = 1
                df.loc[df['complex_signal'] <= self.config.sell_threshold, 'signal'] = -1
            else:
                # 従来の移動平均ベースのシグナル
                df['signal'] = df['ma_signal']
        except Exception as e:
            logger.error(f"複合シグナル計算エラー: {e}")
            # フォールバック: 単純な移動平均シグナル
            df['complex_signal'] = 0
            df['signal'] = df.get('ma_signal', 0)
    
    def calculate_adx(self, data, period=14):
        """
        ADX（平均方向性指数）の計算
        
        Parameters:
        -----------
        data : pandas.DataFrame
            OHLCV データ
        period : int
            ADX計算期間
            
        Returns:
        --------
        float
            ADX値
        """
        try:
            df = data.copy()
            
            # True Range
            df['tr0'] = abs(df['high'] - df['low'])
            df['tr1'] = abs(df['high'] - df['close'].shift())
            df['tr2'] = abs(df['low'] - df['close'].shift())
            df['tr'] = df[['tr0', 'tr1', 'tr2']].max(axis=1)
            
            # +DM, -DM
            df['up_move'] = df['high'] - df['high'].shift()
            df['down_move'] = df['low'].shift() - df['low']
            
            df['plus_dm'] = np.where((df['up_move'] > df['down_move']) & (df['up_move'] > 0), df['up_move'], 0)
            df['minus_dm'] = np.where((df['down_move'] > df['up_move']) & (df['down_move'] > 0), df['down_move'], 0)
            
            # 14期間の平均
            df['plus_di'] = 100 * (df['plus_dm'].rolling(window=period).mean() / df['tr'].rolling(window=period).mean())
            df['minus_di'] = 100 * (df['minus_dm'].rolling(window=period).mean() / df['tr'].rolling(window=period).mean())
            
            # DX
            df['dx'] = 100 * abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di'])
            
            # ADX
            df['adx'] = df['dx'].rolling(window=period).mean()
            
            return df['adx'].iloc[-1] if not df['adx'].empty else 0
        except Exception as e:
            logger.error(f"ADX計算エラー: {e}")
            return 0