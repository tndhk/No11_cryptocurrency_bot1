#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
トレード頻度最適化スクリプト (直接実行版)

使用方法:
    python direct_optimize.py --days 180 --target 15
"""

import os
import argparse
import pandas as pd
import numpy as np
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import sys

# トレーディングボットの必要なコンポーネントをインポート
from config.settings import ConfigManager
from src.data.data_loader import DataLoader
from src.indicators.technical import IndicatorCalculator
from src.signals.signal_generator import SignalGenerator
from src.risk.risk_manager import RiskManager
from src.execution.order_executor import OrderExecutor
from src.backtest.backtester import Backtester
from src.utils.logger import default_logger as logger

def create_components(config):
    """トレーディングボットのコンポーネントを作成"""
    data_loader = DataLoader(config)
    indicator_calculator = IndicatorCalculator(config)
    signal_generator = SignalGenerator(config)
    risk_manager = RiskManager(config)
    
    order_executor = OrderExecutor(config, data_loader, risk_manager)
    backtester = Backtester(config, data_loader, indicator_calculator, 
                           signal_generator, order_executor, risk_manager)
    
    return data_loader, indicator_calculator, signal_generator, risk_manager, order_executor, backtester

def run_backtest_with_params(params, days=30, debug=False):
    """指定されたパラメータでバックテストを実行"""
    # 設定マネージャの作成
    config = ConfigManager()
    
    # パラメータを適用
    for key, value in params.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    # バックテスト日数を設定
    config.backtest_days = days
    
    # コンポーネントを作成
    _, _, _, _, _, backtester = create_components(config)
    
    # フィルター変更が必要な場合
    if params.get("change_filter") == "1":
        modify_signal_generator(True)
    
    # バックテスト実行
    try:
        # ログ出力を一時的に抑制（オプション）
        if not debug:
            original_stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w')
        
        metrics = backtester.run_backtest(debug_signals=False)
        
        # ログ出力を元に戻す
        if not debug:
            sys.stdout = original_stdout
        
        # フィルターを元に戻す
        if params.get("change_filter") == "1":
            modify_signal_generator(False)
        
        return metrics
    
    except Exception as e:
        print(f"バックテストエラー: {e}")
        
        # フィルターを元に戻す
        if params.get("change_filter") == "1":
            modify_signal_generator(False)
        
        # ログ出力を元に戻す
        if not debug:
            sys.stdout = original_stdout
        
        return None

def modify_signal_generator(enable_relaxed=True):
    """シグナルフィルターを修正"""
    filepath = "src/signals/signal_generator.py"
    
    # バックアップがなければ作成
    if not os.path.exists(f"{filepath}.bak"):
        with open(filepath, 'r') as f:
            original_code = f.read()
        
        with open(f"{filepath}.bak", 'w') as f:
            f.write(original_code)
    
    if enable_relaxed:
        # 緩和されたフィルター条件を適用
        with open(filepath, 'r') as f:
            code = f.read()
        
        # トレンド一致度の要件を緩和
        modified_code = code.replace(
            "            # MACDとMAは一致必須\n            if trend_agreement < 2:\n                return 0",
            "            # MACDとMAの一致要件を緩和\n            if trend_agreement < 1:  # 1つの指標だけでOK\n                return 0"
        )
        
        # RSIフィルターを緩和
        modified_code = modified_code.replace(
            "            # RSIが中間帯より下にあることを確認（買いのスペース）\n            if signal_info['rsi'] > 60:\n                return 0",
            "            # RSIフィルターを緩和\n            if signal_info['rsi'] > 70:  # 60から70に変更\n                return 0"
        )
        
        modified_code = modified_code.replace(
            "            # RSIが中間帯より上にあることを確認（売りのスペース）\n            if signal_info['rsi'] < 40:\n                return 0",
            "            # RSIフィルターを緩和\n            if signal_info['rsi'] < 30:  # 40から30に変更\n                return 0"
        )
        
        # 修正したコードを保存
        with open(filepath, 'w') as f:
            f.write(modified_code)
        
        print("シグナルフィルターを緩和しました")
    
    else:
        # 元のコードに戻す
        if os.path.exists(f"{filepath}.bak"):
            with open(f"{filepath}.bak", 'r') as f:
                original_code = f.read()
            
            with open(filepath, 'w') as f:
                f.write(original_code)
            
            print("シグナルフィルターを元に戻しました")

def grid_search_for_frequency(target_trades, days=30, debug=False):
    """目標取引回数に達するパラメータを探索"""
    print(f"{days}日間で目標取引回数 {target_trades} のパラメータを探索中...")
    
    # 探索するパラメータと範囲
    param_grid = {
        # シグナル閾値（最も重要）
        "buy_threshold": [0.5, 0.4, 0.3, 0.2],
        "sell_threshold": [-0.5, -0.4, -0.3, -0.2],
        
        # フィルター緩和用パラメータ
        "rsi_oversold": [30, 35, 40],
        "rsi_overbought": [70, 65, 60],
    }
    
    results = []
    best_config = None
    closest_trades = 0
    best_diff = float('inf')
    
    # グリッドサーチ
    for buy_threshold in param_grid["buy_threshold"]:
        for sell_threshold in param_grid["sell_threshold"]:
            for rsi_oversold in param_grid["rsi_oversold"]:
                for rsi_overbought in param_grid["rsi_overbought"]:
                    params = {
                        "buy_threshold": buy_threshold,
                        "sell_threshold": sell_threshold,
                        "rsi_oversold": rsi_oversold,
                        "rsi_overbought": rsi_overbought,
                        "change_filter": "0"
                    }
                    
                    # バックテスト実行
                    print(f"テスト中: {params}")
                    metrics = run_backtest_with_params(params, days, debug)
                    
                    if metrics:
                        # 結果を記録
                        result = {**params, **metrics}
                        results.append(result)
                        
                        # 取引数を取得
                        total_trades = metrics.get("total_trades", 0)
                        
                        # 目標取引回数との差
                        trades_diff = abs(total_trades - target_trades)
                        
                        # より良い結果が見つかった場合は更新
                        if trades_diff < best_diff:
                            best_diff = trades_diff
                            closest_trades = total_trades
                            best_config = params.copy()
                            
                            print(f"新しい最良構成: {params}")
                            print(f"取引数: {total_trades}, 利益率: {metrics.get('profit_percent', 0):.2f}%, 勝率: {metrics.get('win_rate', 0):.2f}%")
    
    # フィルターコード変更を試す
    if not best_config or abs(closest_trades - target_trades) > 5:
        print("フィルター条件の緩和を試みます...")
        
        # 最も有望なパラメータでフィルター緩和を試す
        if best_config:
            params = best_config.copy()
            params["change_filter"] = "1"
            
            metrics = run_backtest_with_params(params, days, debug)
            if metrics:
                result = {**params, **metrics}
                results.append(result)
                
                # 取引数を取得
                total_trades = metrics.get("total_trades", 0)
                
                trades_diff = abs(total_trades - target_trades)
                if trades_diff < best_diff:
                    best_diff = trades_diff
                    closest_trades = total_trades
                    best_config = params.copy()
                    
                    print(f"フィルター緩和で改善: {params}")
                    print(f"取引数: {total_trades}, 利益率: {metrics.get('profit_percent', 0):.2f}%, 勝率: {metrics.get('win_rate', 0):.2f}%")
    
    # 結果をDataFrameに変換
    results_df = pd.DataFrame(results) if results else pd.DataFrame()
    
    # 結果を保存
    if not results_df.empty:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_df.to_csv(f"results/trade_frequency_optimization_{timestamp}.csv", index=False)
        
        if best_config:
            with open(f"results/best_frequency_params_{timestamp}.json", 'w') as f:
                json.dump(best_config, f, indent=4)
        
        # 結果を可視化
        visualize_results(results_df, timestamp)
    
    return best_config, results_df

def visualize_results(results_df, timestamp):
    """最適化結果の可視化"""
    if len(results_df) < 2:  # 結果が1つだけの場合はスキップ
        print("可視化するには十分なデータがありません")
        return
    
    try:
        plt.figure(figsize=(15, 10))
        
        # 取引数と利益率の関係
        plt.subplot(2, 2, 1)
        plt.scatter(results_df['total_trades'], results_df['profit_percent'])
        plt.title('取引数と利益率の関係')
        plt.xlabel('取引数')
        plt.ylabel('利益率 (%)')
        plt.grid(True)
        
        # 買いシグナル閾値と取引数の関係
        if len(results_df['buy_threshold'].unique()) > 1:
            plt.subplot(2, 2, 2)
            sns.boxplot(x='buy_threshold', y='total_trades', data=results_df)
            plt.title('買いシグナル閾値と取引数')
            plt.xlabel('買いシグナル閾値')
            plt.ylabel('取引数')
        
        # 売りシグナル閾値と取引数の関係
        if len(results_df['sell_threshold'].unique()) > 1:
            plt.subplot(2, 2, 3)
            sns.boxplot(x='sell_threshold', y='total_trades', data=results_df)
            plt.title('売りシグナル閾値と取引数')
            plt.xlabel('売りシグナル閾値')
            plt.ylabel('取引数')
        
        # 取引数と勝率の関係
        plt.subplot(2, 2, 4)
        plt.scatter(results_df['total_trades'], results_df['win_rate'])
        plt.title('取引数と勝率の関係')
        plt.xlabel('取引数')
        plt.ylabel('勝率 (%)')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f"results/trade_frequency_analysis_{timestamp}.png")
        
        print(f"分析結果を保存しました: results/trade_frequency_analysis_{timestamp}.png")
    
    except Exception as e:
        print(f"グラフ作成エラー: {e}")

def quick_test(days=30):
    """迅速なテストを実行して推奨パラメータを確認"""
    print(f"{days}日間のデータで迅速なテストを実行します...")
    
    # 基本設定のテスト
    base_metrics = run_backtest_with_params({}, days, True)
    if base_metrics:
        print("\n基本設定の結果:")
        print(f"取引数: {base_metrics.get('total_trades', 0)}")
        print(f"利益率: {base_metrics.get('profit_percent', 0):.2f}%")
        print(f"勝率: {base_metrics.get('win_rate', 0):.2f}%")
    
    # 閾値を下げたテスト
    lower_threshold_params = {
        "buy_threshold": 0.3,
        "sell_threshold": -0.3
    }
    
    lt_metrics = run_backtest_with_params(lower_threshold_params, days, True)
    if lt_metrics:
        print("\n閾値を下げた結果:")
        print(f"取引数: {lt_metrics.get('total_trades', 0)}")
        print(f"利益率: {lt_metrics.get('profit_percent', 0):.2f}%")
        print(f"勝率: {lt_metrics.get('win_rate', 0):.2f}%")
    
    # フィルターを緩和したテスト
    relaxed_filter_params = {
        "change_filter": "1"
    }
    
    rf_metrics = run_backtest_with_params(relaxed_filter_params, days, True)
    if rf_metrics:
        print("\nフィルターを緩和した結果:")
        print(f"取引数: {rf_metrics.get('total_trades', 0)}")
        print(f"利益率: {rf_metrics.get('profit_percent', 0):.2f}%")
        print(f"勝率: {rf_metrics.get('win_rate', 0):.2f}%")
    
    # 両方を組み合わせたテスト
    combined_params = {
        "buy_threshold": 0.3,
        "sell_threshold": -0.3,
        "change_filter": "1"
    }
    
    combined_metrics = run_backtest_with_params(combined_params, days, True)
    if combined_metrics:
        print("\n閾値を下げ、フィルターも緩和した結果:")
        print(f"取引数: {combined_metrics.get('total_trades', 0)}")
        print(f"利益率: {combined_metrics.get('profit_percent', 0):.2f}%")
        print(f"勝率: {combined_metrics.get('win_rate', 0):.2f}%")
    
    return {
        "base": base_metrics,
        "lower_threshold": lt_metrics,
        "relaxed_filter": rf_metrics,
        "combined": combined_metrics
    }

def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description='トレード頻度最適化スクリプト')
    parser.add_argument('--days', type=int, default=60, help='バックテスト期間（日数）')
    parser.add_argument('--target', type=int, default=15, help='目標取引回数')
    parser.add_argument('--debug', action='store_true', help='デバッグ情報を表示')
    parser.add_argument('--quick', action='store_true', help='迅速なテストを実行')
    args = parser.parse_args()
    
    try:
        if args.quick:
            # 迅速なテストを実行
            quick_test(args.days)
        else:
            # パラメータ探索の実行
            best_config, results_df = grid_search_for_frequency(args.target, args.days, args.debug)
            
            if best_config:
                print("\n最適なパラメータ構成:")
                for key, value in best_config.items():
                    print(f"  {key}: {value}")
                
                # フィルター変更が必要な場合
                if best_config.get("change_filter") == "1":
                    print("\nより多くの取引を生成するには、シグナルフィルターの緩和が必要です")
                    print("このスクリプトは自動的にフィルターを緩和してテストしました")
                    print("恒久的に変更するには、以下の修正を手動で適用してください:")
                    print("""
1. src/signals/signal_generator.py の apply_advanced_filters メソッドで:
   - MACDとMAの一致要件を「if trend_agreement < 1:」に変更
   - RSI制限を「if signal_info['rsi'] > 70:」と「if signal_info['rsi'] < 30:」に変更
                    """)
            else:
                print("\n最適なパラメータ構成が見つかりませんでした")
                print("より広いパラメータ範囲で探索するか、目標取引回数を調整してください")
    
    except KeyboardInterrupt:
        print("\n最適化を中断しました")
    
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()