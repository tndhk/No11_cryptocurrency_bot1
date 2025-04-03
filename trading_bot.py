#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
高度に改良されたトレーディングボット - リファクタリング版

実行方法:
    バックテスト: python trading_bot.py --mode backtest
    ライブトレード: python trading_bot.py --mode live
    最適化: python trading_bot.py --mode optimize
"""

import os
import argparse
import traceback
import warnings

from config.settings import ConfigManager
from src.data.data_loader import DataLoader
from src.indicators.technical import IndicatorCalculator
from src.signals.signal_generator import SignalGenerator
from src.risk.risk_manager import RiskManager
from src.execution.order_executor import OrderExecutor
from src.backtest.backtester import Backtester
from src.optimization.optimizer import ParameterOptimizer
# 以下の行をコメントアウトし、代わりにSimpleOptimizerをインポート
# from src.optimization.advanced_optimizer import AdvancedOptimizer
from src.optimization.simple_optimizer import SimpleOptimizer
from src.utils.logger import default_logger as logger

# 警告を抑制
warnings.filterwarnings('ignore')

def create_components(config):
    """
    コンポーネントを作成

    Parameters:
    -----------
    config : ConfigManager
        設定マネージャ

    Returns:
    --------
    tuple
        (data_loader, indicator_calculator, signal_generator, risk_manager, order_executor, 
         backtester, optimizer, simple_optimizer)
    """
    # 各コンポーネントを初期化
    data_loader = DataLoader(config)
    indicator_calculator = IndicatorCalculator(config)
    signal_generator = SignalGenerator(config)
    risk_manager = RiskManager(config)
    
    # 依存関係を持つコンポーネント
    order_executor = OrderExecutor(config, data_loader, risk_manager)
    backtester = Backtester(config, data_loader, indicator_calculator, 
                           signal_generator, order_executor, risk_manager)
    optimizer = ParameterOptimizer(config, backtester)
    simple_optimizer = SimpleOptimizer(config, backtester)
    
    return (data_loader, indicator_calculator, signal_generator, 
            risk_manager, order_executor, backtester, optimizer, simple_optimizer)

def run_backtest(config, backtester):
    """
    バックテスト実行

    Parameters:
    -----------
    config : ConfigManager
        設定マネージャ
    backtester : Backtester
        バックテスター
    """
    logger.info(f"バックテスト実行: {config.symbol} / {config.interval} / {config.backtest_days}日間")
    
    try:
        metrics = backtester.run_backtest()
        if not metrics:
            logger.error("バックテスト実行中にエラーが発生しました")
    except Exception as e:
        logger.error(f"バックテストエラー: {e}")
        logger.error(traceback.format_exc())

def run_optimization(config, optimizer, simple_optimizer):
    """
    最適化実行

    Parameters:
    -----------
    config : ConfigManager
        設定マネージャ
    optimizer : ParameterOptimizer
        従来のパラメータ最適化
    simple_optimizer : SimpleOptimizer
        シンプルな最適化
    """
    logger.info(f"パラメータ最適化実行: {config.symbol} / {config.interval} / {config.backtest_days}日間")
    
    try:
        # 最適化モードを取得
        optimization_mode = os.getenv("OPTIMIZATION_MODE", "simple").lower()
        
        if optimization_mode == "simple":
            # シンプルな最適化を実行
            logger.info("シンプルな最適化を実行...")
            
            # 重要なパラメータのみを最適化
            if os.getenv("OPTIMIZE_IMPORTANT_ONLY", "true").lower() == "true":
                iterations = int(os.getenv("OPTIMIZATION_ITERATIONS", "50"))
                threshold = float(os.getenv("IMPORTANCE_THRESHOLD", "0.1"))
                
                logger.info(f"重要なパラメータのみを最適化 (閾値: {threshold}, イテレーション: {iterations})")
                best_params = simple_optimizer.optimize_with_importance(
                    importance_threshold=threshold,
                    iterations=iterations
                )
            else:
                # 段階的な最適化
                iterations = int(os.getenv("OPTIMIZATION_ITERATIONS", "50"))
                
                logger.info(f"段階的な最適化を実行 (イテレーション: {iterations})")
                best_params = simple_optimizer.optimize(
                    iterations=iterations
                )
        
        else:
            # 従来のグリッドサーチを実行
            logger.info("従来のグリッドサーチによる最適化を実行...")
            
            # デフォルトのパラメータグリッド
            param_grid = {
                'short_window': [5, 8, 12, 15],
                'long_window': [18, 21, 26, 30],
                'stop_loss_percent': [1.0, 1.5, 2.0, 2.5, 3.0],
                'take_profit_percent': [2.0, 3.0, 5.0, 7.0, 10.0],
                'weight_ma': [0.2, 0.3, 0.4, 0.5],
                'weight_rsi': [0.0, 0.1, 0.2, 0.3],
                'weight_macd': [0.1, 0.2, 0.3, 0.4],
                'weight_bb': [0.0, 0.1, 0.2, 0.3],
                'weight_breakout': [0.0, 0.1, 0.2]
            }
            
            best_params = optimizer.optimize_parameters(param_grid)
        
        if not best_params:
            logger.error("最適化実行中にエラーが発生しました")
        else:
            logger.info("最適化が完了しました。結果は 'results' ディレクトリに保存されています")
    except Exception as e:
        logger.error(f"最適化エラー: {e}")
        logger.error(traceback.format_exc())

def run_live_trading(config, data_loader, indicator_calculator, signal_generator, risk_manager, order_executor):
    """
    ライブトレード実行

    Parameters:
    -----------
    config : ConfigManager
        設定マネージャ
    その他コンポーネント
    """
    logger.info(f"ライブトレード実行: {config.symbol} / {config.interval}")
    
    try:
        # ライブモードでは長時間実行が前提なので、
        # 通常はスケジューラなどを使って定期的に実行します
        # ここでは簡略化のため、一度だけの実行を行います
        
        # 最新のデータを取得
        data = data_loader.get_historical_data(limit=100)
        
        if data.empty:
            logger.error("市場データの取得に失敗しました")
            return
        
        # 指標計算
        df = indicator_calculator.calculate_indicators(data)
        
        # シグナル生成
        signal_info = signal_generator.generate_trading_signals(df)
        
        # 取引実行（バックテストではなくライブモード）
        order_executor.execute_trade(signal_info, is_backtest=False, signal_generator=signal_generator)
        
        logger.info("ライブトレード実行を完了しました")
        
    except Exception as e:
        logger.error(f"ライブトレードエラー: {e}")
        logger.error(traceback.format_exc())

def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description='高度なBinanceトレーディングボット')
    parser.add_argument('--mode', choices=['backtest', 'live', 'optimize', 'analyze'], default='backtest',
                        help='実行モード: backtest, live, optimize, analyze')
    args = parser.parse_args()
    
    # 設定マネージャの初期化
    config = ConfigManager()
    
    # 各コンポーネントを作成
    data_loader, indicator_calculator, signal_generator, \
    risk_manager, order_executor, backtester, optimizer, simple_optimizer = create_components(config)
    
    # 実行モードに応じた処理
    if args.mode == 'backtest':
        run_backtest(config, backtester)
    elif args.mode == 'live':
        run_live_trading(config, data_loader, indicator_calculator, 
                        signal_generator, risk_manager, order_executor)
    elif args.mode == 'optimize':
        run_optimization(config, optimizer, simple_optimizer)
    elif args.mode == 'analyze':
        # パラメータ重要度分析のみを実行
        logger.info("パラメータ重要度分析を実行...")
        importance = simple_optimizer.get_param_importance(
            num_iterations=int(os.getenv("IMPORTANCE_ITERATIONS", "30"))
        )
        if importance:
            logger.info("分析が完了しました。結果は 'results' ディレクトリに保存されています")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"メインプロセスエラー: {e}")
        logger.error(traceback.format_exc())