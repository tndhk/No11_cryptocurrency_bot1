from src.utils.metrics import calculate_drawdown

class RiskManager:
    """
    リスク管理クラス
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
        
        # リスク管理の状態
        self.consecutive_losses = 0
        self.current_balance = 10000  # バックテスト初期残高
        self.initial_trade_quantity = float(config.trade_quantity)
        self.trade_quantity = self.initial_trade_quantity
        self.peak_balance = 10000
    
    def update_balance(self, balance):
        """
        残高情報を更新
        
        Parameters:
        -----------
        balance : float
            最新の口座残高
        """
        self.current_balance = balance
        self.peak_balance = max(self.peak_balance, balance)
    
    def update_trade_result(self, profit):
        """
        取引結果に基づいて連続損失カウンタを更新
        
        Parameters:
        -----------
        profit : float
            取引の純利益
        """
        if profit <= 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0  # 利益が出たらリセット
    
    def check_risk_limits(self, balance_history):
        """
        リスク制限をチェック
        
        Parameters:
        -----------
        balance_history : list
            残高履歴のリスト [(timestamp, balance), ...]
            
        Returns:
        --------
        bool
            取引続行可能か
        str
            制限メッセージ（制限なしの場合はNone）
        """
        # 連続損失チェック
        if self.consecutive_losses >= self.config.max_consecutive_losses:
            # 取引サイズを半分に減らす
            if self.trade_quantity > self.initial_trade_quantity / 4:  # 最小サイズまで
                self.trade_quantity = self.trade_quantity / 2
                return True, f"連続損失により取引サイズを削減: {self.trade_quantity}"
        
        # 最大ドローダウン制限
        current_drawdown = calculate_drawdown(balance_history)
        if current_drawdown > self.config.max_drawdown_limit:
            # 取引を一時停止
            return False, f"最大ドローダウン制限({self.config.max_drawdown_limit}%)に達したため取引を一時停止"
        
        return True, None  # 制限なし
    
    def calculate_position_size(self, entry_price, stop_loss_price):
        """
        リスクに基づくポジションサイズの計算
        
        Parameters:
        -----------
        entry_price : float
            エントリー価格
        stop_loss_price : float
            ストップロス価格
            
        Returns:
        --------
        float
            取引数量
        """
        # リスク計算：資本の1%をリスクに
        risk_amount = self.current_balance * self.config.risk_per_trade
        
        # 価格あたりのリスク（ストップロスまでの距離）
        price_risk = abs(entry_price - stop_loss_price)
        
        if price_risk == 0:
            return self.trade_quantity  # デフォルト値
        
        # 許容リスクに基づく数量計算
        quantity = risk_amount / price_risk
        
        # 最小・最大取引サイズの制限
        min_trade = 0.0005
        max_trade = 0.005
        
        return max(min(quantity, max_trade), min_trade)
    
    def calculate_dynamic_risk_reward(self, signal_info):
        """
        市場状況に基づいた動的なリスク/リワード設定
        
        Parameters:
        -----------
        signal_info : dict
            シグナル情報
            
        Returns:
        --------
        tuple
            (stop_loss_percent, take_profit_percent)
        """
        # 基本設定を拡大（より大きなリワード）
        sl_percent = self.config.stop_loss_percent
        tp_percent = self.config.take_profit_percent * 1.5  # 1.5倍に増加
        
        # トレンド状況に応じた調整
        if signal_info['ma_signal'] == signal_info['macd_signal']:
            # トレンド一致度が高い場合はさらにリワード拡大
            tp_percent *= 1.2
        
        # ボラティリティに基づく調整
        atr_ratio = signal_info.get('atr', 0) / signal_info['close']
        if atr_ratio > 0.01:  # 高ボラティリティ
            # 高ボラティリティ時はより広いTPとSL
            tp_percent *= 1.2
            sl_percent *= 1.2
        
        return sl_percent, tp_percent