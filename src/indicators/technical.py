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
        # 単純移動平均（SMA）
        df['SMA_short'] = df['close'].rolling(window=self.config.short_window).mean()
        df['SMA_long'] = df['close'].rolling(window=self.config.long_window).mean()
        
        # 指数移動平均（EMA）
        df['EMA_short'] = df['close'].ewm(span=self.config.short_window, adjust=False).mean()
        df['EMA_long'] = df['close'].ewm(span=self.config.long_window, adjust=False).mean()
        
        # 移動平均クロスオーバーシグナルの計算
        df['ma_signal'] = 0
        df.loc[df['EMA_short'] > df['EMA_long'], 'ma_signal'] = 1
        df.loc[df['EMA_short'] < df['EMA_long'], 'ma_signal'] = -1
    
    def _calculate_rsi(self, df):
        """RSIを計算"""
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=self.config.rsi_period).mean()
        avg_loss = loss.rolling(window=self.config.rsi_period).mean()
        
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # RSIベースのシグナル
        df['rsi_signal'] = 0
        df.loc[df['RSI'] < self.config.rsi_oversold, 'rsi_signal'] = 1  # 買い
        df.loc[df['RSI'] > self.config.rsi_overbought, 'rsi_signal'] = -1  # 売り
    
    def _calculate_bollinger_bands(self, df):
        """ボリンジャーバンドを計算"""
        df['BB_middle'] = df['close'].rolling(window=self.config.bb_period).mean()
        df['BB_std'] = df['close'].rolling(window=self.config.bb_period).std()
        df['BB_upper'] = df['BB_middle'] + self.config.bb_std * df['BB_std']
        df['BB_lower'] = df['BB_middle'] - self.config.bb_std * df['BB_std']
        
        # ボリンジャーバンドベースのシグナル
        df['bb_signal'] = 0
        df.loc[df['close'] < df['BB_lower'], 'bb_signal'] = 1  # 買い（下限突破）
        df.loc[df['close'] > df['BB_upper'], 'bb_signal'] = -1  # 売り（上限突破）
    
    def _calculate_macd(self, df):
        """MACDを計算"""
        df['EMA_' + str(self.config.macd_fast)] = df['close'].ewm(span=self.config.macd_fast, adjust=False).mean()
        df['EMA_' + str(self.config.macd_slow)] = df['close'].ewm(span=self.config.macd_slow, adjust=False).mean()
        df['MACD'] = df['EMA_' + str(self.config.macd_fast)] - df['EMA_' + str(self.config.macd_slow)]
        df['MACD_signal'] = df['MACD'].ewm(span=self.config.macd_signal, adjust=False).mean()
        df['MACD_hist'] = df['MACD'] - df['MACD_signal']
        
        # MACDベースのシグナル
        df['macd_signal'] = 0
        df.loc[(df['MACD'] > df['MACD_signal']) & (df['MACD'].shift(1) <= df['MACD_signal'].shift(1)), 'macd_signal'] = 1  # 買い（MACD上抜け）
        df.loc[(df['MACD'] < df['MACD_signal']) & (df['MACD'].shift(1) >= df['MACD_signal'].shift(1)), 'macd_signal'] = -1  # 売り（MACD下抜け）
    
    def _calculate_breakout(self, df):
        """ブレイクアウトを計算"""
        n_periods = 14
        df['highest_high'] = df['high'].rolling(window=n_periods).max()
        df['lowest_low'] = df['low'].rolling(window=n_periods).min()
        
        # ブレイクアウトベースのシグナル
        df['breakout_signal'] = 0
        df.loc[df['close'] > df['highest_high'].shift(1), 'breakout_signal'] = 1  # 買い（高値ブレイク）
        df.loc[df['close'] < df['lowest_low'].shift(1), 'breakout_signal'] = -1  # 売り（安値ブレイク）
    
    def _calculate_volume_indicators(self, df):
        """出来高指標を計算"""
        df['volume_change'] = df['volume'].pct_change()
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        
        # 出来高ベースのシグナル
        df['volume_signal'] = 0
        df.loc[(df['volume'] > df['volume_ma'] * 1.5) & (df['close'] > df['open']), 'volume_signal'] = 1  # 大出来高の上昇
        df.loc[(df['volume'] > df['volume_ma'] * 1.5) & (df['close'] < df['open']), 'volume_signal'] = -1  # 大出来高の下落
    
    def _calculate_atr(self, df):
        """ATR（Average True Range）を計算"""
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['ATR'] = tr.rolling(14).mean()
    
    def _calculate_complex_signal(self, df):
        """複合シグナルを計算"""
        if self.config.use_complex_signal:
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