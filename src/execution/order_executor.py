import numpy as np
import traceback
from datetime import datetime
from binance.client import Client
from src.utils.logger import default_logger as logger

class OrderExecutor:
    """
    注文実行とシミュレーションクラス
    """
    
    def __init__(self, config, data_loader, risk_manager):
        """
        初期化
        
        Parameters:
        -----------
        config: ConfigManager
            設定マネージャのインスタンス
        data_loader: DataLoader
            データローダのインスタンス
        risk_manager: RiskManager
            リスク管理のインスタンス
        """
        self.config = config
        self.data_loader = data_loader
        self.risk_manager = risk_manager
        
        # 取引状態
        self.in_position = False
        self.entry_price = 0
        self.stop_loss = 0
        self.take_profit = 0
        self.current_trade = {}
        
        # 取引履歴
        self.trades = []
    
    def execute_trade(self, signal_info, is_backtest=False, signal_generator=None):
        """
        取引を実行
        
        Parameters:
        -----------
        signal_info : dict
            シグナル情報
        is_backtest : bool
            バックテストモードかどうか
        signal_generator : SignalGenerator, optional
            シグナル生成クラス（理由取得用）
        """
        signal = signal_info['signal']
        
        # リスク管理制限のチェック
        can_trade, risk_message = self.risk_manager.check_risk_limits(self.get_balance_history())
        if not can_trade:
            logger.warning(risk_message)
            return
        elif risk_message:
            logger.info(risk_message)
        
        if signal == 1 and not self.in_position:
            # 買いシグナルかつポジションなし
            self._execute_buy(signal_info, is_backtest, signal_generator)
                
        elif signal == -1 and self.in_position:
            # 売りシグナルかつポジションあり
            self._execute_sell(signal_info, is_backtest, signal_generator)
    
    def _execute_buy(self, signal_info, is_backtest, signal_generator):
        """買い注文を実行"""
        try:
            # 実行価格のシミュレーション
            execution_price = self.simulate_execution_price(signal_info, is_buy=True)

            # 動的ポジションサイズ計算
            temp_stop_loss = execution_price * (1 - self.config.stop_loss_percent/100)
            position_size = self.risk_manager.calculate_position_size(execution_price, temp_stop_loss)
            # 取引サイズを更新
            self.risk_manager.trade_quantity = position_size
            logger.info(f"動的ポジションサイズ: {position_size}")

            if not is_backtest:
                self.data_loader._initialize_client()
                self.data_loader._check_api_rate_limit()
                # 実際の取引実行
                order = self.data_loader.client.create_order(
                    symbol=self.config.symbol,
                    side=Client.SIDE_BUY,
                    type=Client.ORDER_TYPE_MARKET,
                    quantity=position_size
                )
                order_id = order['orderId']
            else:
                # バックテスト用の仮想注文ID
                order_id = f"backtest_buy_{datetime.now().timestamp()}"
            
            self.in_position = True
            self.entry_price = execution_price

            # 動的リスク/リワード比の適用
            sl_percent, tp_percent = self.risk_manager.calculate_dynamic_risk_reward(signal_info)
            self.stop_loss = execution_price * (1 - sl_percent/100)
            self.take_profit = execution_price * (1 + tp_percent/100)
            
            # 取引情報
            trade_info = {
                'type': 'BUY',
                'timestamp': signal_info['timestamp'],
                'signal_price': signal_info['close'],
                'execution_price': execution_price,
                'slippage_percent': (execution_price / signal_info['close'] - 1) * 100,
                'quantity': position_size,
                'reason': signal_generator.get_signal_reason(signal_info) if signal_generator else "Signal",
                'order_id': order_id,
                'stop_loss': self.stop_loss,
                'take_profit': self.take_profit,
                'indicators': {
                    'rsi': signal_info['rsi'],
                    'macd': signal_info['macd'],
                    'sma_short': signal_info['sma_short'],
                    'sma_long': signal_info['sma_long'],
                    'complex_signal': signal_info.get('complex_signal', 0)
                }
            }
            self.trades.append(trade_info)
            self.current_trade = trade_info
            
            logger.info(f"買い注文: {self.config.symbol} @ {execution_price:.2f} USDT (シグナル価格: {signal_info['close']:.2f})")
            logger.info(f"スリッページ: {trade_info['slippage_percent']:.2f}%")
            logger.info(f"ストップロス: {self.stop_loss:.2f} / テイクプロフィット: {self.take_profit:.2f}")
            
        except Exception as e:
            logger.error(f"買い注文エラー: {e}")
            logger.error(traceback.format_exc())
    
    def _execute_sell(self, signal_info, is_backtest, signal_generator):
        """売り注文を実行"""
        try:
            # 実行価格のシミュレーション
            execution_price = self.simulate_execution_price(signal_info, is_buy=False)
            
            if not is_backtest:
                self.data_loader._initialize_client()
                self.data_loader._check_api_rate_limit()
                # 実際の取引実行
                order = self.data_loader.client.create_order(
                    symbol=self.config.symbol,
                    side=Client.SIDE_SELL,
                    type=Client.ORDER_TYPE_MARKET,
                    quantity=self.risk_manager.trade_quantity
                )
                order_id = order['orderId']
            else:
                # バックテスト用の仮想注文ID
                order_id = f"backtest_sell_{datetime.now().timestamp()}"
            
            # 手数料を考慮した利益計算
            gross_profit = (execution_price - self.entry_price) * self.risk_manager.trade_quantity
            net_profit = gross_profit * (1 - self.config.taker_fee)  # 手数料を差し引く
            
            # 損益によって連続損失カウンターを更新
            self.risk_manager.update_trade_result(net_profit)
            
            # バランス更新
            self.risk_manager.update_balance(self.risk_manager.current_balance + net_profit)

            # 利益率計算
            profit_percent = (execution_price / self.entry_price - 1) * 100
            
            trade_info = {
                'type': 'SELL',
                'timestamp': signal_info['timestamp'],
                'signal_price': signal_info['close'],
                'execution_price': execution_price,
                'slippage_percent': (execution_price / signal_info['close'] - 1) * 100,
                'quantity': self.risk_manager.trade_quantity,
                'gross_profit': gross_profit,
                'net_profit': net_profit,  # 手数料を考慮した純利益
                'profit_percent': profit_percent,
                'fees': gross_profit * self.config.taker_fee,
                'reason': signal_generator.get_signal_reason(signal_info) if signal_generator else "Signal",
                'order_id': order_id,
                'entry_price': self.entry_price,
                'hold_duration': self.calculate_hold_duration(self.current_trade['timestamp'], signal_info['timestamp']),
                'indicators': {
                    'rsi': signal_info['rsi'],
                    'macd': signal_info['macd'],
                    'sma_short': signal_info['sma_short'],
                    'sma_long': signal_info['sma_long'],
                    'complex_signal': signal_info.get('complex_signal', 0)
                }
            }
            self.trades.append(trade_info)
            
            logger.info(f"売り注文: {self.config.symbol} @ {execution_price:.2f} USDT (シグナル価格: {signal_info['close']:.2f})")
            logger.info(f"スリッページ: {trade_info['slippage_percent']:.2f}%")
            logger.info(f"純利益: {net_profit:.4f} USDT ({profit_percent:.2f}%)")
            
            self.in_position = False
            self.current_trade = {}
            
        except Exception as e:
            logger.error(f"売り注文エラー: {e}")
            logger.error(traceback.format_exc())
    
    def calculate_slippage(self, is_buy):
        """
        スリッページをシミュレート
        
        Parameters:
        -----------
        is_buy : bool
            買い注文かどうか
            
        Returns:
        --------
        float
            スリッページ率（正の値は価格上昇、負の値は価格下落）
        """
        # 正規分布に基づくランダムなスリッページ
        random_slippage = np.random.normal(self.config.slippage_mean, self.config.slippage_std)
        
        # 買い注文の場合は正のスリッページ（価格上昇）、売り注文の場合は負のスリッページ（価格下落）
        if is_buy:
            return abs(random_slippage)  # 買いは高くなる（不利）
        else:
            return -abs(random_slippage)  # 売りは安くなる（不利）
    
    def simulate_execution_price(self, signal_info, is_buy):
        """
        実行価格をシミュレート
        
        Parameters:
        -----------
        signal_info : dict
            シグナル情報
        is_buy : bool
            買い注文かどうか
            
        Returns:
        --------
        float
            実行価格
        """
        # シグナル発生時の終値
        close_price = signal_info['close']
        
        # 次の足の始値をシミュレート（現在の終値に±1%のランダム変動を加える）
        next_open_percent = np.random.normal(0, 0.01)  # 平均0、標準偏差1%
        next_open = close_price * (1 + next_open_percent)
        
        # スリッページを加算
        slippage_percent = self.calculate_slippage(is_buy)
        execution_price = next_open * (1 + slippage_percent)
        
        return execution_price
    
    def simulate_detailed_price_path(self, open_price, high_price, low_price, close_price, num_steps=None):
        """
        ローソク足内での詳細な価格パスをシミュレーション
        
        Parameters:
        -----------
        open_price, high_price, low_price, close_price : float
            ローソク足のOHLC値
        num_steps : int, optional
            シミュレーションするステップ数（精度）
            
        Returns:
        --------
        list
            時間的に整列された価格系列
        """
        if num_steps is None:
            num_steps = self.config.price_simulation_steps
            
        # ローソク足の方向性
        is_bullish = close_price > open_price
        
        # 価格変動の振れ幅
        price_range = high_price - low_price
        
        # ブラウン運動に基づく価格パスの生成
        price_path = []
        current_price = open_price
        
        # ランダムウォークに加えて、終値に向かうトレンド成分を追加
        trend_strength = 0.6  # 終値へ引き寄せる力の強さ (0-1)
        
        for i in range(num_steps):
            # 現在のステップの進捗率 (0-1)
            progress = i / (num_steps - 1)
            
            # ランダム成分（ボラティリティ）
            random_component = np.random.normal(0, price_range * 0.03)
            
            # トレンド成分（終値へ向かう力）
            trend_component = (close_price - current_price) * trend_strength * progress
            
            # 価格更新
            current_price += random_component + trend_component
            
            # 高値・安値の範囲内に制約
            current_price = max(min(current_price, high_price * 1.001), low_price * 0.999)
            
            price_path.append(current_price)
        
        # 最後の価格は必ず終値に一致させる
        price_path[-1] = close_price
        
        return price_path
    
    def check_sl_tp_on_price_path(self, price_path, stop_loss, take_profit):
        """
        価格パス上でのSL/TP発動を検出
        
        Parameters:
        -----------
        price_path : list
            時系列順の価格パス
        stop_loss : float
            ストップロス価格
        take_profit : float
            テイクプロフィット価格
            
        Returns:
        --------
        tuple
            (exit_type, exit_price, exit_index)
            exit_typeは 'Stop Loss', 'Take Profit', None のいずれか
        """
        for i, price in enumerate(price_path):
            if price <= stop_loss:
                return 'Stop Loss', price, i
            if price >= take_profit:
                return 'Take Profit', price, i
        
        return None, None, None
    
    def simulate_intracandle_execution(self, candle_data, stop_loss, take_profit):
        """
        ローソク足内での価格変動を使ってストップロスとテイクプロフィットをシミュレート
        
        Parameters:
        -----------
        candle_data : dict or pd.Series
            ローソク足データ (open, high, low, close)
        stop_loss : float
            ストップロス価格
        take_profit : float
            テイクプロフィット価格
            
        Returns:
        --------
        tuple
            (exit_reason, exit_price)
            exit_reasonは 'Stop Loss', 'Take Profit', None のいずれか
        """
        # 価格パスが無効な場合はシミュレーションしない
        if not self.config.use_price_simulation:
            # 単純にローソク足の高値と安値を使って判定
            if candle_data['low'] <= stop_loss:
                return 'Stop Loss', stop_loss
            elif candle_data['high'] >= take_profit:
                return 'Take Profit', take_profit
            return None, None
            
        # ローソク足内での詳細な価格パスをシミュレート
        price_path = self.simulate_detailed_price_path(
            candle_data['open'], 
            candle_data['high'], 
            candle_data['low'], 
            candle_data['close'],
            self.config.price_simulation_steps
        )
        
        # 価格パス上でのSL/TP発動を検出
        exit_type, exit_price, _ = self.check_sl_tp_on_price_path(price_path, stop_loss, take_profit)
        
        return exit_type, exit_price
    
    def calculate_hold_duration(self, start_time, end_time):
        """
        ポジションの保有期間を計算（時間単位）
        
        Parameters:
        -----------
        start_time : datetime or pd.Timestamp
            エントリー時間
        end_time : datetime or pd.Timestamp
            決済時間
            
        Returns:
        --------
        float
            保有期間（時間）
        """
        try:
            duration = (end_time - start_time).total_seconds() / 3600  # 時間に変換
            return round(duration, 1)
        except:
            return 0
    
    def get_trades(self):
        """取引履歴を取得"""
        return self.trades
    
    def get_balance_history(self):
        """残高履歴を取得"""
        # この実装では不完全かもしれない（バックテスター側で実装）
        return []