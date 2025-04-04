import pandas as pd
from src.utils.logger import default_logger as logger

class SignalGenerator:
    """
    シグナル生成クラス
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
    
    def generate_trading_signals(self, data):
        """
        トレーディングシグナルの生成
        
        Parameters:
        -----------
        data : pandas.DataFrame
            テクニカル指標が計算されたデータフレーム
            
        Returns:
        --------
        dict
            シグナル情報
        """
        if data.empty or len(data) < 2:
            return {}
                
        # 最新のデータポイント
        current = data.iloc[-1]
        previous = data.iloc[-2]
        
        # 基本シグナル
        signal = 0
        
        # シグナル変化を検出
        if previous['signal'] != 1 and current['signal'] == 1:
            signal = 1  # 買いシグナル
        elif previous['signal'] != -1 and current['signal'] == -1:
            signal = -1  # 売りシグナル
        
        # シグナル情報をまとめる
        signal_info = {
            'timestamp': current['timestamp'],
            'open': current['open'],
            'high': current['high'],
            'low': current['low'],
            'close': current['close'],
            'signal': signal,
            'ma_signal': current['ma_signal'],
            'rsi_signal': current['rsi_signal'],
            'bb_signal': current.get('bb_signal', 0),
            'macd_signal': current.get('macd_signal', 0),
            'breakout_signal': current.get('breakout_signal', 0),
            'complex_signal': current.get('complex_signal', 0),
            'sma_short': current['SMA_short'],
            'sma_long': current['SMA_long'],
            'ema_short': current['EMA_short'],
            'ema_long': current['EMA_long'],
            'rsi': current['RSI'],
            'macd': current.get('MACD', 0),
            'macd_signal_line': current.get('MACD_signal', 0),
            'bb_upper': current.get('BB_upper', 0),
            'bb_lower': current.get('BB_lower', 0),
            'atr': current.get('ATR', 0)
        }
        
        # シグナルが発生した場合にはフィルター適用
        if signal != 0:
            # フィルタリングでシグナルの品質を向上
            filtered_signal = self.apply_advanced_filters(signal_info)
            signal_info['signal'] = filtered_signal
        
        return signal_info
    
    def apply_advanced_filters(self, signal_info):
        """
        シグナル品質向上フィルター
        
        Parameters:
        -----------
        signal_info : dict
            シグナル情報
            
        Returns:
        --------
        int
            フィルタリング後のシグナル値
        """
        signal = signal_info['signal']
        
        # トレンド一致度の確認
        trend_agreement = 0
        if signal > 0:  # 買いシグナル
            if signal_info['ma_signal'] > 0: trend_agreement += 1
            if signal_info['macd_signal'] > 0: trend_agreement += 1
            
            # MACDとMAは一致必須
            if trend_agreement < 2:
                return 0
                
            # RSIが中間帯より下にあることを確認（買いのスペース）
            if signal_info['rsi'] > 60:
                return 0
                
        elif signal < 0:  # 売りシグナル
            if signal_info['ma_signal'] < 0: trend_agreement += 1
            if signal_info['macd_signal'] < 0: trend_agreement += 1
            
            # MACDとMAは一致必須
            if trend_agreement < 2:
                return 0
                
            # RSIが中間帯より上にあることを確認（売りのスペース）
            if signal_info['rsi'] < 40:
                return 0
        
        return signal
    
    def generate_multi_timeframe_signals(self, data_h4, data_h1):
        """
        複数の時間枠からのシグナルを統合
        
        Parameters:
        -----------
        data_h4 : pandas.DataFrame
            4時間足データ
        data_h1 : pandas.DataFrame
            1時間足データ
            
        Returns:
        --------
        int
            統合シグナル (1 = 買い, -1 = 売り, 0 = 中立)
        """
        # 両方のデータフレームが有効かチェック
        if data_h4.empty or data_h1.empty:
            return 0
        
        # 各時間枠の最新シグナル
        h4_signal = data_h4.iloc[-1]['signal']
        h1_signal = data_h1.iloc[-1]['signal']
        
        # シグナルの統合（例：両方がポジティブなら強いシグナル）
        if h4_signal == 1 and h1_signal == 1:
            return 1  # 強い買いシグナル
        elif h4_signal == -1 and h1_signal == -1:
            return -1  # 強い売りシグナル
        
        return 0  # 中立シグナル
    
    def adapt_parameters_to_market(self, data, indicator_calculator):
        """
        市場ボラティリティに基づいてパラメータを調整
        
        Parameters:
        -----------
        data : pandas.DataFrame
            市場データ
        indicator_calculator : IndicatorCalculator
            指標計算オブジェクト
        """
        # 最近のボラティリティを計算
        recent_data = data.tail(50)
        volatility = recent_data['close'].pct_change().std() * 100
        
        # ボラティリティに基づいて閾値を調整
        if volatility > 3.0:  # 高ボラティリティ
            self.config.buy_threshold = 0.4
            self.config.sell_threshold = -0.4
            logger.info(f"高ボラティリティ検出: {volatility:.2f}% - 閾値を調整")
        elif volatility > 1.5:  # 中ボラティリティ
            self.config.buy_threshold = 0.3
            self.config.sell_threshold = -0.3
            logger.info(f"中ボラティリティ検出: {volatility:.2f}% - 閾値を調整")
        else:  # 低ボラティリティ
            self.config.buy_threshold = 0.2
            self.config.sell_threshold = -0.2
            logger.info(f"低ボラティリティ検出: {volatility:.2f}% - 閾値を調整")
            
        # トレンド強度に基づいた重み付け調整
        adx = indicator_calculator.calculate_adx(data)
        if adx > 25:  # 強いトレンド
            self.config.weight_ma = 0.45
            self.config.weight_macd = 0.35
            self.config.weight_rsi = 0.1
            self.config.weight_bb = 0.1
            logger.info(f"強いトレンド検出 (ADX: {adx:.2f}) - 重みを調整")
        else:  # 弱いトレンド/レンジ相場
            self.config.weight_ma = 0.3
            self.config.weight_macd = 0.2
            self.config.weight_rsi = 0.2
            self.config.weight_bb = 0.3
            logger.info(f"弱いトレンド検出 (ADX: {adx:.2f}) - 重みを調整")
    
    def get_signal_reason(self, signal_info):
        """
        シグナルの理由を判断
        
        Parameters:
        -----------
        signal_info : dict
            シグナル情報
            
        Returns:
        --------
        str
            シグナルの理由
        """
        reasons = []
        
        if signal_info.get('ma_signal', 0) != 0:
            reasons.append('移動平均')
        
        if signal_info.get('rsi_signal', 0) != 0:
            reasons.append('RSI')
        
        if signal_info.get('macd_signal', 0) != 0:
            reasons.append('MACD')
        
        if signal_info.get('bb_signal', 0) != 0:
            reasons.append('ボリンジャーバンド')
        
        if signal_info.get('breakout_signal', 0) != 0:
            reasons.append('ブレイクアウト')
        
        if not reasons:
            return '複合シグナル'
        
        return '+'.join(reasons)