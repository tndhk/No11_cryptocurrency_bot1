import time
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import PercentFormatter
import seaborn as sns
import json
from datetime import datetime
import traceback

from src.utils.logger import default_logger as logger
from src.utils.metrics import calculate_performance_metrics

class Backtester:
    """
    バックテスト実行クラス
    """
    
    def __init__(self, config, data_loader, indicator_calculator, signal_generator, order_executor, risk_manager):
        """
        初期化
        
        Parameters:
        -----------
        config: ConfigManager
            設定マネージャのインスタンス
        data_loader: DataLoader
            データローダのインスタンス
        indicator_calculator: IndicatorCalculator
            指標計算クラスのインスタンス
        signal_generator: SignalGenerator
            シグナル生成クラスのインスタンス
        order_executor: OrderExecutor
            注文実行クラスのインスタンス
        risk_manager: RiskManager
            リスク管理クラスのインスタンス
        """
        self.config = config
        self.data_loader = data_loader
        self.indicator_calculator = indicator_calculator
        self.signal_generator = signal_generator
        self.order_executor = order_executor
        self.risk_manager = risk_manager
        
        # バックテスト用の状態
        self.balance_history = []
        self.debug_signals = []
    
    def run_backtest(self, debug_signals=True, early_stopping=False, min_trades=5, profit_threshold=-10.0):
        """
        バックテストを実行する
        
        Parameters:
        -----------
        debug_signals : bool
            シグナルのデバッグ情報を収集するかどうか
        early_stopping : bool
            早期終了機能を使用するかどうか
        min_trades : int
            早期終了判断のための最小トレード数
        profit_threshold : float
            早期終了する利益率の閾値（これより悪い場合に終了）
        
        Returns:
        --------
        dict
            バックテスト結果
        """
        logger.info("バックテストモードを開始")
        
        # 過去データの取得
        end_time = int(time.time() * 1000)
        start_time = end_time - (self.config.backtest_days * 24 * 60 * 60 * 1000)
        
        try:
            # 効率的なデータ取得
            data = self.data_loader.get_historical_data(
                start_time=start_time, 
                end_time=end_time, 
                is_backtest=True
            )
            
            if data.empty:
                logger.error("バックテスト用データが取得できませんでした")
                return {}
            
            logger.info(f"取得データ: {len(data)} ロウソク足 ({data['timestamp'].min()} - {data['timestamp'].max()})")
            
            # 指標計算
            df = self.indicator_calculator.calculate_indicators(data)
            
            # 初期資本
            initial_balance = 10000  # USDT
            balance = initial_balance
            self.order_executor.in_position = False
            self.order_executor.trades = []
            self.balance_history = [(df.iloc[0]['timestamp'], balance)]
            self.debug_signals = []
            
            # データポイントが十分かチェック
            min_required_points = max(self.config.long_window, self.config.macd_slow) + 5
            if len(df) <= min_required_points:
                logger.error(f"バックテスト用データが不足しています（必要: {min_required_points}, 取得: {len(df)}）")
                return {}

            # バックテスト実行
            for i in range(min_required_points, len(df) - 1):
                prev_data = df.iloc[:i+1]
                current_data = df.iloc[i]
                signal_info = self.signal_generator.generate_trading_signals(prev_data)
                current_price = current_data['close']

                # シグナル生成後にデバッグ情報を収集
                if debug_signals and signal_info.get('complex_signal', 0) != 0:
                    self.debug_signals.append({
                        'timestamp': signal_info['timestamp'],
                        'price': current_price,
                        'complex_signal': signal_info.get('complex_signal', 0),
                        'ma_signal': signal_info['ma_signal'],
                        'rsi_signal': signal_info['rsi_signal'],
                        'macd_signal': signal_info['macd_signal'],
                        'bb_signal': signal_info.get('bb_signal', 0),
                        'breakout_signal': signal_info.get('breakout_signal', 0),
                    })

                # ポジションがある場合、ローソク足内でのSL/TPチェック
                if self.order_executor.in_position:
                    # ローソク足内での価格変動をシミュレート
                    exit_reason, exit_price = self.order_executor.simulate_intracandle_execution(
                        current_data, self.order_executor.stop_loss, self.order_executor.take_profit
                    )
                    
                    if exit_reason:
                        # スリッページを適用
                        slippage = self.order_executor.calculate_slippage(is_buy=False)
                        execution_price = exit_price * (1 + slippage)
                        
                        # 手数料を考慮した利益計算
                        gross_profit = (execution_price - self.order_executor.entry_price) * self.risk_manager.trade_quantity
                        fees = gross_profit * self.config.taker_fee
                        net_profit = gross_profit - fees
                        
                        # 残高に純利益を加算
                        balance += net_profit
                        
                        # 損益によって連続損失カウンターを更新
                        self.risk_manager.update_trade_result(net_profit)
                        
                        # 取引情報
                        self.order_executor.trades.append({
                            'type': 'SELL',
                            'timestamp': current_data['timestamp'],
                            'signal_price': current_price,
                            'execution_price': execution_price,
                            'quantity': self.risk_manager.trade_quantity,
                            'gross_profit': gross_profit,
                            'net_profit': net_profit,
                            'fees': fees,
                            'profit_percent': (execution_price / self.order_executor.entry_price - 1) * 100,
                            'reason': exit_reason,
                            'slippage_percent': (execution_price / exit_price - 1) * 100,
                            'entry_price': self.order_executor.entry_price,
                            'hold_duration': self.order_executor.calculate_hold_duration(
                                self.order_executor.current_trade.get('timestamp', current_data['timestamp']), 
                                current_data['timestamp']
                            )
                        })
                        
                        self.order_executor.in_position = False
                        self.order_executor.current_trade = {}
                
                # デイトレードでは通常、次のローソク足の始値でエントリー
                next_candle_idx = i + self.config.execution_delay
                if next_candle_idx < len(df):
                    next_candle = df.iloc[next_candle_idx]
                    
                    # シグナルに基づく取引
                    if signal_info['signal'] == 1 and not self.order_executor.in_position:
                        # 実行価格をシミュレート（次の足の始値＋スリッページ）
                        execution_price = float(next_candle['open'])
                        slippage = self.order_executor.calculate_slippage(is_buy=True)
                        execution_price *= (1 + slippage)
                        
                        # 買い注文を実行
                        self.order_executor.entry_price = execution_price
                        
                        # 動的リスク/リワード比の適用
                        sl_percent, tp_percent = self.risk_manager.calculate_dynamic_risk_reward(signal_info)
                        self.order_executor.stop_loss = execution_price * (1 - sl_percent/100)
                        self.order_executor.take_profit = execution_price * (1 + tp_percent/100)
                        
                        self.order_executor.in_position = True
                        
                        # 取引手数料を適用
                        execution_cost = execution_price * self.risk_manager.trade_quantity
                        fees = execution_cost * self.config.taker_fee
                        
                        # トレード情報
                        trade_info = {
                            'type': 'BUY',
                            'timestamp': next_candle['timestamp'],
                            'signal_timestamp': df.iloc[i]['timestamp'],
                            'signal_price': current_price,
                            'execution_price': execution_price,
                            'quantity': self.risk_manager.trade_quantity,
                            'slippage_percent': (execution_price / float(next_candle['open']) - 1) * 100,
                            'fees': fees,
                            'reason': self.signal_generator.get_signal_reason(signal_info),
                            'stop_loss': self.order_executor.stop_loss,
                            'take_profit': self.order_executor.take_profit
                        }
                        
                        self.order_executor.trades.append(trade_info)
                        
                        # 現在の取引情報を更新
                        self.order_executor.current_trade = trade_info
                        
                    elif signal_info['signal'] == -1 and self.order_executor.in_position:
                        # 実行価格をシミュレート（次の足の始値＋スリッページ）
                        execution_price = float(next_candle['open'])
                        slippage = self.order_executor.calculate_slippage(is_buy=False)
                        execution_price *= (1 + slippage)
                        
                        # 手数料を考慮した利益計算
                        gross_profit = (execution_price - self.order_executor.entry_price) * self.risk_manager.trade_quantity
                        fees = gross_profit * self.config.taker_fee
                        net_profit = gross_profit - fees
                        
                        # 残高に純利益を加算
                        balance += net_profit
                        
                        # 損益によって連続損失カウンターを更新
                        self.risk_manager.update_trade_result(net_profit)
                        
                        self.order_executor.trades.append({
                            'type': 'SELL',
                            'timestamp': next_candle['timestamp'],
                            'signal_timestamp': df.iloc[i]['timestamp'],
                            'signal_price': current_price,
                            'execution_price': execution_price,
                            'quantity': self.risk_manager.trade_quantity,
                            'gross_profit': gross_profit,
                            'net_profit': net_profit,
                            'fees': fees,
                            'profit_percent': (execution_price / self.order_executor.entry_price - 1) * 100,
                            'reason': self.signal_generator.get_signal_reason(signal_info),
                            'slippage_percent': (execution_price / float(next_candle['open']) - 1) * 100,
                            'entry_price': self.order_executor.entry_price,
                            'hold_duration': self.order_executor.calculate_hold_duration(
                                self.order_executor.current_trade.get('timestamp', df.iloc[i]['timestamp']), 
                                next_candle['timestamp']
                            )
                        })
                        
                        self.order_executor.in_position = False
                        self.order_executor.current_trade = {}
                
                # 残高履歴を更新
                self.balance_history.append((current_data['timestamp'], balance))
                self.risk_manager.current_balance = balance  # 追加
                self.risk_manager.peak_balance = max(self.risk_manager.peak_balance, balance)  # 追加
                
                # 早期終了判断
                if early_stopping and len(self.order_executor.trades) >= min_trades:
                    current_profit = (balance / initial_balance - 1) * 100
                    if current_profit < profit_threshold:
                        logger.debug(f"早期終了: 現在の利益率 {current_profit:.2f}% が閾値 {profit_threshold:.2f}% を下回りました")
                        # メトリクスを計算して早期終了
                        from src.utils.metrics import calculate_performance_metrics
                        return calculate_performance_metrics(
                            self.order_executor.trades, 
                            self.balance_history, 
                            initial_balance
                        )
            
            # 最後のポジションがまだ残っている場合、クローズ
            if self.order_executor.in_position:
                last_price = df.iloc[-1]['close']
                gross_profit = (last_price - self.order_executor.entry_price) * self.risk_manager.trade_quantity
                fees = gross_profit * self.config.taker_fee
                net_profit = gross_profit - fees
                
                balance += net_profit
                
                self.order_executor.trades.append({
                    'type': 'SELL',
                    'timestamp': df.iloc[-1]['timestamp'],
                    'execution_price': last_price,
                    'quantity': self.risk_manager.trade_quantity,
                    'gross_profit': gross_profit,
                    'net_profit': net_profit,
                    'fees': fees,
                    'profit_percent': (last_price / self.order_executor.entry_price - 1) * 100,
                    'reason': 'End of Backtest',
                    'entry_price': self.order_executor.entry_price,
                    'hold_duration': self.order_executor.calculate_hold_duration(
                        self.order_executor.current_trade.get('timestamp', df.iloc[-1]['timestamp']), 
                        df.iloc[-1]['timestamp']
                    )
                })
                
                self.order_executor.in_position = False
                self.order_executor.current_trade = {}
                
                # 最終残高を更新
                self.balance_history.append((df.iloc[-1]['timestamp'], balance))
            
            # バックテスト結果
            self.plot_backtest_results(df)
            metrics = self.print_backtest_summary(initial_balance, balance)
            
            # デバッグシグナル情報の出力
            if debug_signals and self.debug_signals:
                logger.info(f"生成されたシグナル数: {len(self.debug_signals)}")
                logger.info(f"シグナル強度の平均: {sum([s['complex_signal'] for s in self.debug_signals])/len(self.debug_signals):.4f}")
                logger.info(f"最大シグナル強度: {max([s['complex_signal'] for s in self.debug_signals]):.4f}")
                logger.info(f"最小シグナル強度: {min([s['complex_signal'] for s in self.debug_signals]):.4f}")
            elif debug_signals:
                logger.warning("シグナルが生成されませんでした。閾値または重みの調整が必要です。")
            
            return metrics
            
        except Exception as e:
            logger.error(f"バックテストエラー: {e}")
            logger.error(traceback.format_exc())
            return {}

    def plot_backtest_results(self, data):
        """
        バックテスト結果のグラフ作成
        
        Parameters:
        -----------
        data : pandas.DataFrame
            テクニカル指標データ
        """
        try:
            # タイムスタンプ（プロットのファイル名用）
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # データの準備
            df = data.copy()
            
            # 取引情報からデータフレームを作成
            trades_df = pd.DataFrame(self.order_executor.trades)
            buy_trades = trades_df[trades_df['type'] == 'BUY']
            sell_trades = trades_df[trades_df['type'] == 'SELL']
            
            # 残高履歴をデータフレームに変換
            balance_df = pd.DataFrame(self.balance_history, columns=['timestamp', 'balance'])
            
            # 結果を4つのサブプロットで表示
            plt.figure(figsize=(16, 20))
            
            # 1. 価格チャートと取引ポイント
            plt.subplot(4, 1, 1)
            plt.plot(df['timestamp'], df['close'], label='Close Price', alpha=0.7)
            plt.plot(df['timestamp'], df['SMA_short'], label=f'SMA {self.config.short_window}', alpha=0.5)
            plt.plot(df['timestamp'], df['SMA_long'], label=f'SMA {self.config.long_window}', alpha=0.5)
            
            if not buy_trades.empty:
                plt.scatter(buy_trades['timestamp'], buy_trades['execution_price'], marker='^', color='g', s=100, label='Buy')
            if not sell_trades.empty:
                plt.scatter(sell_trades['timestamp'], sell_trades['execution_price'], marker='v', color='r', s=100, label='Sell')
            
            plt.title('Price Chart with Trading Points')
            plt.xlabel('Date')
            plt.ylabel('Price (USDT)')
            plt.grid(True)
            plt.legend()
            
            # 2. 残高推移
            plt.subplot(4, 1, 2)
            plt.plot(balance_df['timestamp'], balance_df['balance'], label='Balance', color='blue')
            plt.title('Balance History')
            plt.xlabel('Date')
            plt.ylabel('Balance (USDT)')
            plt.grid(True)
            
            # 3. RSIとMACDの推移
            plt.subplot(4, 1, 3)
            plt.plot(df['timestamp'], df['RSI'], label='RSI', color='purple')
            plt.axhline(y=70, color='r', linestyle='--', alpha=0.3)
            plt.axhline(y=30, color='g', linestyle='--', alpha=0.3)
            plt.title('RSI Indicator')
            plt.xlabel('Date')
            plt.ylabel('RSI')
            plt.grid(True)
            plt.legend()
            
            # 4. MACD
            plt.subplot(4, 1, 4)
            plt.bar(df['timestamp'], df['MACD_hist'], label='MACD Histogram', color='gray', alpha=0.3)
            plt.plot(df['timestamp'], df['MACD'], label='MACD', color='blue')
            plt.plot(df['timestamp'], df['MACD_signal'], label='Signal Line', color='red')
            plt.title('MACD Indicator')
            plt.xlabel('Date')
            plt.ylabel('MACD')
            plt.grid(True)
            plt.legend()
            
            # グラフの保存
            plt.tight_layout()
            plt.savefig(f'results/backtest_{self.config.symbol}_{timestamp}.png')
            plt.close()
            
            # 追加のグラフ: 取引結果の分析
            self._plot_trade_analysis(trades_df, timestamp)
            
            logger.info(f"バックテスト結果グラフを保存しました: results/backtest_{self.config.symbol}_{timestamp}.png")
            
        except Exception as e:
            logger.error(f"グラフ作成エラー: {e}")
            logger.error(traceback.format_exc())
    
    def _plot_trade_analysis(self, trades_df, timestamp):
        """取引結果の詳細分析グラフを作成"""
        if trades_df.empty or 'profit_percent' not in trades_df.columns:
            return
        
        sell_trades = trades_df[trades_df['type'] == 'SELL'].copy()
        if sell_trades.empty:
            return
        
        try:
            plt.figure(figsize=(15, 15))
            
            # 1. 取引の利益分布
            plt.subplot(3, 2, 1)
            sns.histplot(sell_trades['profit_percent'], bins=20, kde=True)
            plt.axvline(x=0, color='r', linestyle='--')
            plt.title('Profit Distribution (%)')
            plt.xlabel('Profit %')
            
            # 2. 保有時間と利益の関係
            if 'hold_duration' in sell_trades.columns:
                plt.subplot(3, 2, 2)
                plt.scatter(sell_trades['hold_duration'], sell_trades['profit_percent'])
                plt.title('Hold Duration vs Profit')
                plt.xlabel('Hold Duration (hours)')
                plt.ylabel('Profit %')
                plt.grid(True)
            
            # 3. 取引理由別の利益
            if 'reason' in sell_trades.columns:
                plt.subplot(3, 2, 3)
                reason_profit = sell_trades.groupby('reason')['profit_percent'].mean().sort_values()
                reason_profit.plot(kind='barh')
                plt.title('Average Profit by Exit Reason')
                plt.xlabel('Average Profit %')
                
                # 4. 取引理由別の回数
                plt.subplot(3, 2, 4)
                reason_count = sell_trades['reason'].value_counts()
                reason_count.plot(kind='barh')
                plt.title('Number of Trades by Exit Reason')
                plt.xlabel('Count')
            
            # 5. 月別収益
            if 'timestamp' in sell_trades.columns:
                plt.subplot(3, 2, 5)
                sell_trades['month'] = sell_trades['timestamp'].dt.strftime('%Y-%m')
                monthly_profit = sell_trades.groupby('month')['net_profit'].sum()
                monthly_profit.plot(kind='bar')
                plt.title('Monthly Profit')
                plt.xlabel('Month')
                plt.ylabel('Profit (USDT)')
                plt.xticks(rotation=45)
            
            # 6. 累積利益
            plt.subplot(3, 2, 6)
            sell_trades = sell_trades.sort_values('timestamp')
            sell_trades['cumulative_profit'] = sell_trades['net_profit'].cumsum()
            plt.plot(sell_trades['timestamp'], sell_trades['cumulative_profit'])
            plt.title('Cumulative Profit')
            plt.xlabel('Date')
            plt.ylabel('Profit (USDT)')
            plt.grid(True)
            
            # グラフの保存
            plt.tight_layout()
            plt.savefig(f'results/trade_analysis_{self.config.symbol}_{timestamp}.png')
            plt.close()
            
            logger.info(f"取引分析グラフを保存しました: results/trade_analysis_{self.config.symbol}_{timestamp}.png")
            
        except Exception as e:
            logger.error(f"取引分析グラフ作成エラー: {e}")
            logger.error(traceback.format_exc())
    
    def print_backtest_summary(self, initial_balance, final_balance):
        """
        バックテスト結果サマリーを出力
        
        Parameters:
        -----------
        initial_balance : float
            初期残高
        final_balance : float
            最終残高
            
        Returns:
        --------
        dict
            バックテスト結果メトリクス
        """
        # メトリクスの計算
        metrics = calculate_performance_metrics(
            self.order_executor.trades, 
            self.balance_history, 
            initial_balance
        )
        
        logger.info("=" * 50)
        logger.info("バックテスト結果")
        logger.info("=" * 50)
        logger.info(f"初期資本: {initial_balance:.2f} USDT")
        logger.info(f"最終資本: {final_balance:.2f} USDT")
        logger.info(f"純利益: {metrics['net_profit']:.2f} USDT ({metrics['profit_percent']:.2f}%)")
        logger.info(f"取引数: {metrics['total_trades']}")
        logger.info(f"勝率: {metrics['win_rate']:.2f}%（{metrics['winning_trades']}勝 / {metrics['closed_trades'] - metrics['winning_trades']}敗）")
        
        if metrics['avg_profit'] > 0:
            logger.info(f"平均利益: {metrics['avg_profit']:.4f} USDT")
        
        if metrics['avg_loss'] < 0:
            logger.info(f"平均損失: {metrics['avg_loss']:.4f} USDT")
        
        logger.info(f"損益比率: {metrics['profit_factor']:.2f}")
        logger.info(f"最大ドローダウン: {metrics['max_drawdown']:.2f}%")
        logger.info(f"平均保有時間: {metrics['avg_hold_time']:.1f} 時間")
        
        logger.info("=" * 50)
        
        # 結果をJSONとして保存
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            with open(f'results/backtest_summary_{self.config.symbol}_{timestamp}.json', 'w') as f:
                json.dump(metrics, f, default=str, indent=4)
        except Exception as e:
            logger.error(f"結果保存エラー: {e}")
        
        return metrics