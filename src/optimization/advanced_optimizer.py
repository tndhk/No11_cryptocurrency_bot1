import json
import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import traceback
import copy
from functools import partial
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

# ベイズ最適化ライブラリ
from skopt import gp_minimize, gbrt_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from skopt.plots import plot_objective, plot_convergence

# 互換性パッチの適用
from src.optimization.compatibility_fix import apply_compatibility_patches
patched_forest_minimize = apply_compatibility_patches()

from src.utils.logger import default_logger as logger

class AdvancedOptimizer:
    """
    高度なパラメータ最適化クラス - ベイズ最適化対応
    """
    
    def __init__(self, config, backtester):
        """
        初期化
        
        Parameters:
        -----------
        config: ConfigManager
            設定マネージャのインスタンス
        backtester: Backtester
            バックテストクラスのインスタンス
        """
        self.config = config
        self.backtester = backtester
        
        # 最適化設定
        self.num_workers = self._determine_worker_count()
        
        # 早期終了設定
        self.early_stop_threshold = -5.0  # -5%以下なら早期終了
        self.min_trades_for_early_stop = 5  # 最低5トレード実行
        
        # 段階的最適化のグループ定義
        self.param_groups = {
            "moving_averages": ["short_window", "long_window"],
            "risk_management": ["stop_loss_percent", "take_profit_percent"],
            "weights": ["weight_ma", "weight_rsi", "weight_macd", "weight_bb", "weight_breakout"],
            "signal_thresholds": ["buy_threshold", "sell_threshold"]
        }
        
        # 進捗追跡
        self.start_time = None
        self.total_iterations = 0
        self.completed_iterations = 0
        
        # 結果保存用
        self.results = []
    
    def _determine_worker_count(self):
        """ワーカー数を決定"""
        if self.config.optimization_workers > 0:
            # 明示的に指定された場合
            return min(self.config.optimization_workers, os.cpu_count())
        else:
            # 自動：コア数 - 1（最低1）
            return max(1, os.cpu_count() - 1)
    
    def optimize(self, iterations=100, method='bayesian'):
        """
        高度な最適化を実行
        
        Parameters:
        -----------
        iterations : int
            各段階の最適化イテレーション数
        method : str
            'bayesian', 'forest', 'gbrt' のいずれか
            
        Returns:
        --------
        dict
            最適なパラメータと結果
        """
        logger.info("高度な段階的最適化を開始...")
        
        # 開始時間の記録
        self.start_time = time.time()
        
        # 元の設定をバックアップ
        original_params = self._backup_config()
        
        try:
            # 段階的最適化の実行
            result = self._run_staged_optimization(iterations, method)
            
            # 結果保存
            self._save_optimization_results(result)
            
            # 最適化結果をログ出力
            self._log_optimization_results(result)
            
            # 元の設定に戻す
            self._restore_config(original_params)
            
            return result
        
        except Exception as e:
            logger.error(f"最適化エラー: {e}")
            logger.error(traceback.format_exc())
            
            # 元の設定に戻す
            self._restore_config(original_params)
            
            return None
    
    def _backup_config(self):
        """現在の設定をバックアップ"""
        params = {}
        for group in self.param_groups.values():
            for param in group:
                if hasattr(self.config, param):
                    params[param] = getattr(self.config, param)
        return params
    
    def _restore_config(self, params):
        """バックアップした設定を復元"""
        for param, value in params.items():
            if hasattr(self.config, param):
                setattr(self.config, param, value)
    
    def _run_staged_optimization(self, iterations, method):
        """段階的最適化を実行"""
        best_params = {}
        
        # 各パラメータグループで順次最適化
        for group_name, params in self.param_groups.items():
            logger.info(f"パラメータグループ '{group_name}' の最適化を開始...")
            
            # 既に最適化したパラメータを適用
            self._apply_params(best_params)
            
            # このグループのパラメータだけを最適化
            group_result = self._optimize_parameter_group(params, iterations, method)
            
            # 結果をマージ
            if group_result:
                for param, value in group_result.items():
                    best_params[param] = value
                
                logger.info(f"グループ '{group_name}' の最適パラメータ: {group_result}")
            else:
                logger.warning(f"グループ '{group_name}' の最適化に失敗しました")
        
        # 最終的な最適パラメータで性能評価
        self._apply_params(best_params)
        final_metrics = self.backtester.run_backtest(debug_signals=False)
        
        result = {**best_params, **final_metrics}
        return result
    
    def _apply_params(self, params):
        """設定にパラメータを適用"""
        for param, value in params.items():
            if hasattr(self.config, param):
                setattr(self.config, param, value)
    
    def _optimize_parameter_group(self, params, iterations, method):
        """
        パラメータグループを最適化
        
        Parameters:
        -----------
        params : list
            最適化するパラメータ名のリスト
        iterations : int
            最適化イテレーション数
        method : str
            最適化手法
            
        Returns:
        --------
        dict
            最適なパラメータ
        """
        # パラメータの探索空間を定義
        space = self._create_search_space(params)
        
        # パラメータがなければスキップ
        if not space:
            return {}
        
        # 目的関数を作成（最小化のため、負の利益を返す）
        @use_named_args(space)
        def objective_function(**kwargs):
            # パラメータを設定
            for param, value in kwargs.items():
                setattr(self.config, param, value)
            
            # バックテスト実行（早期終了機能付き）
            metrics = self._run_backtest_with_early_stopping()
            
            # 進捗を更新
            self.completed_iterations += 1
            self._report_progress()
            
            # 最小化のため、負の利益率を返す
            profit = metrics.get('profit_percent', 0)
            return -profit
        
        # 探索アルゴリズム選択
        optimizer_func = gp_minimize
        if method == 'forest':
            optimizer_func = patched_forest_minimize  # パッチ適用版を使用
        elif method == 'gbrt':
            optimizer_func = gbrt_minimize
        
        # 総イテレーション数の更新
        self.total_iterations += iterations
        
        # 最適化実行
        logger.info(f"{len(params)}個のパラメータの{method}最適化を{iterations}回のイテレーションで実行します")
        
        # 最適化結果
        result = optimizer_func(
            objective_function,
            space,
            n_calls=iterations,
            n_jobs=self.num_workers,  # 並列処理
            verbose=True
        )
        
        # 最適なパラメータを取得
        best_params = {}
        for idx, param in enumerate([dim.name for dim in space]):
            best_params[param] = result.x[idx]
        
        # 最適化グラフを保存
        self._save_optimization_plots(result, params)
        
        return best_params
    
    def _create_search_space(self, params):
        """
        探索空間を作成
        
        Parameters:
        -----------
        params : list
            パラメータ名のリスト
            
        Returns:
        --------
        list
            skoptの探索空間定義
        """
        space = []
        
        param_ranges = {
            # 移動平均設定
            "short_window": Integer(5, 20, name="short_window"),
            "long_window": Integer(20, 50, name="long_window"),
            
            # リスク管理設定
            "stop_loss_percent": Real(1.0, 5.0, name="stop_loss_percent"),
            "take_profit_percent": Real(2.0, 15.0, name="take_profit_percent"),
            
            # 重み設定
            "weight_ma": Real(0.1, 0.5, name="weight_ma"),
            "weight_rsi": Real(0.0, 0.4, name="weight_rsi"),
            "weight_macd": Real(0.0, 0.4, name="weight_macd"),
            "weight_bb": Real(0.0, 0.4, name="weight_bb"),
            "weight_breakout": Real(0.0, 0.3, name="weight_breakout"),
            
            # 閾値設定
            "buy_threshold": Real(0.2, 0.8, name="buy_threshold"),
            "sell_threshold": Real(-0.8, -0.2, name="sell_threshold"),
        }
        
        for param in params:
            if param in param_ranges:
                space.append(param_ranges[param])
            else:
                logger.warning(f"パラメータ '{param}' の探索範囲が定義されていません")
        
        return space
    
    def _run_backtest_with_early_stopping(self):
        """
        早期終了機能付きバックテスト実行
        
        Returns:
        --------
        dict
            バックテスト結果
        """
        # ここで直接バックテスターの内部処理を呼ぶか、
        # バックテスターに早期終了機能を追加して呼び出すかの2つの方法があります
        # 今回は簡単のため、標準のバックテスト関数を呼び出します
        
        # 実際の実装では、バックテスターに early_stopping パラメータを追加し、
        # トレード数と利益率をモニタリングして早期に終了する機能を実装したほうが良いでしょう
        
        metrics = self.backtester.run_backtest(debug_signals=False)
        return metrics
    
    def _report_progress(self):
        """進捗状況を報告"""
        if self.total_iterations <= 0:
            return
        
        # 進捗率
        progress = self.completed_iterations / self.total_iterations * 100
        
        # 経過時間
        elapsed_time = time.time() - self.start_time
        
        # 残り時間の推定
        if self.completed_iterations > 0:
            time_per_iter = elapsed_time / self.completed_iterations
            remaining_time = time_per_iter * (self.total_iterations - self.completed_iterations)
        else:
            remaining_time = 0
        
        # 10%ごとまたは10回ごとに進捗を報告
        if self.completed_iterations % max(1, self.total_iterations // 10) == 0 or self.completed_iterations % 10 == 0:
            logger.info(f"進捗: {progress:.1f}% ({self.completed_iterations}/{self.total_iterations}) " 
                       f"経過時間: {elapsed_time:.1f}秒 残り時間: {remaining_time:.1f}秒")
    
    def _save_optimization_plots(self, result, params):
        """
        最適化結果のグラフを保存
        
        Parameters:
        -----------
        result : OptimizeResult
            skoptの最適化結果
        params : list
            パラメータ名のリスト
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # パラメータグループ名を生成
            group_name = "_".join(params)
            
            # 収束プロット
            plt.figure(figsize=(10, 6))
            plot_convergence(result)
            plt.title(f"最適化収束 - {group_name}")
            plt.tight_layout()
            plt.savefig(f"results/optimize_convergence_{group_name}_{timestamp}.png")
            plt.close()
            
            # パラメータ空間プロット（2つ以上のパラメータがある場合）
            if len(params) >= 2:
                plt.figure(figsize=(15, 10))
                plot_objective(result, n_points=10)
                plt.suptitle(f"パラメータ空間 - {group_name}")
                plt.tight_layout()
                plt.savefig(f"results/optimize_objective_{group_name}_{timestamp}.png")
                plt.close()
            
            logger.info(f"最適化グラフを保存しました: グループ {group_name}")
            
        except Exception as e:
            logger.error(f"グラフ保存エラー: {e}")
    
    def _save_optimization_results(self, result):
        """最適化結果を保存"""
        if not result:
            return
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # JSONとして保存
            with open(f"results/advanced_optimization_{self.config.symbol}_{timestamp}.json", 'w') as f:
                json.dump(result, f, default=str, indent=4)
            
            logger.info(f"最適化結果を保存しました: results/advanced_optimization_{self.config.symbol}_{timestamp}.json")
            
        except Exception as e:
            logger.error(f"結果保存エラー: {e}")
    
    def _log_optimization_results(self, result):
        """最適化結果をログ出力"""
        if not result:
            return
        
        # パラメータグループごとにログ出力
        logger.info("=" * 80)
        logger.info("最適化結果")
        logger.info("=" * 80)
        
        for group_name, params in self.param_groups.items():
            logger.info(f"パラメータグループ '{group_name}':")
            for param in params:
                if param in result:
                    logger.info(f"  {param}: {result[param]}")
        
        logger.info("-" * 80)
        logger.info("パフォーマンス指標:")
        metrics = ["profit_percent", "win_rate", "profit_factor", "max_drawdown", "total_trades"]
        for metric in metrics:
            if metric in result:
                logger.info(f"  {metric}: {result[metric]}")
        
        logger.info("=" * 80)
    
    def get_param_importance(self, num_iterations=50):
        """
        パラメータ重要度分析
        
        Parameters:
        -----------
        num_iterations : int
            重要度分析のイテレーション数
            
        Returns:
        --------
        dict
            各パラメータの重要度
        """
        logger.info("パラメータ重要度分析を開始...")
        
        # すべてのパラメータを取得
        all_params = []
        for group in self.param_groups.values():
            all_params.extend(group)
        
        # 探索空間を定義
        space = self._create_search_space(all_params)
        
        # 既存の設定をバックアップ
        original_params = self._backup_config()
        
        try:
            # ランダムなパラメータでバックテストを実行
            results = []
            
            # 目的関数
            @use_named_args(space)
            def objective_function(**kwargs):
                # パラメータを設定
                for param, value in kwargs.items():
                    setattr(self.config, param, value)
                
                # バックテスト実行
                metrics = self.backtester.run_backtest(debug_signals=False)
                
                # 結果を記録
                result = {**kwargs, **metrics}
                results.append(result)
                
                # 進捗を報告
                logger.info(f"重要度分析: {len(results)}/{num_iterations} 完了")
                
                # 最小化のため、負の利益率を返す
                profit = metrics.get('profit_percent', 0)
                return -profit
            
            # ランダムサンプリングで実行
            from skopt.utils import cook_initial_point_generator
            initial_points_generator = cook_initial_point_generator(
                "random", xs=None, random_state=None
            )
            
            # 最適化実行（ランダムサンプリング）- パッチ適用版を使用
            result = patched_forest_minimize(
                objective_function,
                space,
                n_calls=num_iterations,
                n_jobs=self.num_workers,
                initial_point_generator=initial_points_generator,
                verbose=True
            )
            
            # 結果をDataFrameに変換
            if results:
                results_df = pd.DataFrame(results)
                
                # 相関分析
                correlation = results_df[all_params + ['profit_percent']].corr()['profit_percent']
                importance = correlation.abs().sort_values(ascending=False)
                
                # 重要度グラフ
                plt.figure(figsize=(12, 8))
                importance.drop('profit_percent').plot(kind='bar')
                plt.title('パラメータ重要度 (利益との相関の絶対値)')
                plt.tight_layout()
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                plt.savefig(f"results/param_importance_{self.config.symbol}_{timestamp}.png")
                plt.close()
                
                # ヒートマップ
                plt.figure(figsize=(15, 12))
                sns.heatmap(
                    results_df[all_params + ['profit_percent']].corr(), 
                    annot=True, 
                    cmap='coolwarm',
                    vmin=-1, 
                    vmax=1
                )
                plt.title('パラメータ間の相関')
                plt.tight_layout()
                plt.savefig(f"results/param_correlation_{self.config.symbol}_{timestamp}.png")
                plt.close()
                
                logger.info("パラメータ重要度:")
                for param, imp in importance.drop('profit_percent').items():
                    logger.info(f"  {param}: {imp:.4f}")
                
                return importance.drop('profit_percent').to_dict()
            else:
                logger.warning("重要度分析のための有効な結果がありません")
                return {}
                
        except Exception as e:
            logger.error(f"パラメータ重要度分析エラー: {e}")
            logger.error(traceback.format_exc())
            return {}
            
        finally:
            # 元の設定に戻す
            self._restore_config(original_params)
    
    def optimize_with_importance(self, importance_threshold=0.1, iterations=50):
        """
        重要なパラメータのみを最適化
        
        Parameters:
        -----------
        importance_threshold : float
            重要と判断する相関係数の閾値
        iterations : int
            最終最適化のイテレーション数
            
        Returns:
        --------
        dict
            最適なパラメータと結果
        """
        logger.info(f"重要なパラメータのみの最適化を開始 (閾値: {importance_threshold})...")
        
        # 重要度分析（少ないイテレーション数）
        importance = self.get_param_importance(num_iterations=20)
        
        if not importance:
            logger.warning("重要度分析に失敗したため、通常の最適化を実行します")
            return self.optimize(iterations)
        
        # 重要なパラメータのみ選択
        important_params = [param for param, imp in importance.items() if abs(imp) >= importance_threshold]
        
        if not important_params:
            logger.warning(f"重要なパラメータが見つかりませんでした (閾値: {importance_threshold})")
            logger.warning("すべてのパラメータで最適化を実行します")
            return self.optimize(iterations)
        
        logger.info(f"重要なパラメータ ({len(important_params)}個): {important_params}")
        
        # 重要なパラメータのみで最適化
        space = self._create_search_space(important_params)
        
        # 元の設定をバックアップ
        original_params = self._backup_config()
        
        try:
            # 目的関数
            @use_named_args(space)
            def objective_function(**kwargs):
                # パラメータを設定
                for param, value in kwargs.items():
                    setattr(self.config, param, value)
                
                # バックテスト実行
                metrics = self.backtester.run_backtest(debug_signals=False)
                
                # 進捗を更新
                self._report_progress()
                
                # 最小化のため、負の利益率を返す
                profit = metrics.get('profit_percent', 0)
                return -profit
            
            # 総イテレーション数の更新
            self.total_iterations = iterations
            self.completed_iterations = 0
            self.start_time = time.time()
            
            # 最適化実行
            result = gp_minimize(
                objective_function,
                space,
                n_calls=iterations,
                n_jobs=self.num_workers,
                verbose=True
            )
            
            # 最適なパラメータを取得
            best_params = {}
            for idx, param in enumerate([dim.name for dim in space]):
                best_params[param] = result.x[idx]
            
            # 最終的な性能評価
            self._apply_params(best_params)
            final_metrics = self.backtester.run_backtest(debug_signals=False)
            
            # 結果をマージ
            result_dict = {**best_params, **final_metrics}
            
            # 結果を保存
            self._save_optimization_results(result_dict)
            
            # 最適化グラフを保存
            self._save_optimization_plots(result, important_params)
            
            # 結果をログ出力
            self._log_optimization_results(result_dict)
            
            return result_dict
            
        except Exception as e:
            logger.error(f"重要パラメータ最適化エラー: {e}")
            logger.error(traceback.format_exc())
            return None
            
        finally:
            # 元の設定に戻す
            self._restore_config(original_params)