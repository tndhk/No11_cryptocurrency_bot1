"""
シンプルなベイズ最適化モジュール

scikit-optimizeを直接使わずにシンプルなベイズ最適化機能を提供します。
これにより依存関係の問題を回避します。
"""

import os
import time
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import traceback
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor, as_completed

from src.utils.logger import default_logger as logger


class SimpleOptimizer:
    """
    シンプルなパラメータ最適化クラス
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
    
    def optimize(self, iterations=50):
        """
        最適化を実行
        
        Parameters:
        -----------
        iterations : int
            各パラメータグループの最適化イテレーション数
            
        Returns:
        --------
        dict
            最適なパラメータと結果
        """
        logger.info("段階的最適化を開始...")
        
        # 開始時間の記録
        self.start_time = time.time()
        
        # 元の設定をバックアップ
        original_params = self._backup_config()
        
        try:
            # 段階的最適化の実行
            best_params = {}
            
            # 各パラメータグループを順次最適化
            for group_name, params in self.param_groups.items():
                logger.info(f"パラメータグループ '{group_name}' の最適化を開始...")
                
                # 既に最適化したパラメータを適用
                self._apply_params(best_params)
                
                # このグループのパラメータだけを最適化
                group_result = self._optimize_parameter_group(params, iterations)
                
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
            
            # 結果をマージ
            result = {**best_params, **final_metrics}
            
            # 結果を保存
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
    
    def _apply_params(self, params):
        """設定にパラメータを適用"""
        for param, value in params.items():
            if hasattr(self.config, param):
                setattr(self.config, param, value)
    
    def _optimize_parameter_group(self, params, iterations):
        """
        パラメータグループを最適化
        
        Parameters:
        -----------
        params : list
            最適化するパラメータ名のリスト
        iterations : int
            最適化イテレーション数
            
        Returns:
        --------
        dict
            最適なパラメータ
        """
        # パラメータの探索空間を定義
        param_ranges = self._create_param_ranges(params)
        
        # パラメータがなければスキップ
        if not param_ranges:
            return {}
        
        # ランダムサンプリングの準備
        self.total_iterations = iterations
        self.completed_iterations = 0
        
        # 結果を格納する配列
        results = []
        
        try:
            # 並列処理でランダムサンプリング
            with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                # イテレーションごとにタスクを投入
                futures = []
                for i in range(iterations):
                    # ランダムなパラメータ組み合わせを生成
                    params_dict = self._generate_random_params(param_ranges)
                    # タスク投入
                    futures.append(executor.submit(self._evaluate_params, params_dict, i))
                
                # 結果を収集
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        if result:
                            results.append(result)
                            # 進捗報告
                            self.completed_iterations += 1
                            self._report_progress()
                    except Exception as e:
                        logger.error(f"評価タスクエラー: {e}")
            
            # 結果が空の場合
            if not results:
                logger.warning("有効な結果がありませんでした")
                return {}
            
            # 結果をDataFrameに変換
            results_df = pd.DataFrame(results)
            
            # 最良の結果を抽出
            best_result = results_df.loc[results_df['profit_percent'].idxmax()]
            
            # パラメータだけを抽出
            best_params = {param: best_result[param] for param in params if param in best_result}
            
            # 結果をグラフ化（オプション）
            self._plot_optimization_results(results_df, params)
            
            return best_params
            
        except Exception as e:
            logger.error(f"最適化実行エラー: {e}")
            logger.error(traceback.format_exc())
            return {}
    
    def _create_param_ranges(self, params):
        """
        探索範囲を定義
        
        Parameters:
        -----------
        params : list
            パラメータ名のリスト
            
        Returns:
        --------
        dict
            パラメータ名とその探索範囲のマッピング
        """
        param_ranges = {}
        
        # 各パラメータの探索範囲を定義
        for param in params:
            if param == "short_window":
                param_ranges[param] = {'min': 5, 'max': 20, 'type': 'int'}
            elif param == "long_window":
                param_ranges[param] = {'min': 20, 'max': 50, 'type': 'int'}
            elif param == "stop_loss_percent":
                param_ranges[param] = {'min': 1.0, 'max': 5.0, 'type': 'float'}
            elif param == "take_profit_percent":
                param_ranges[param] = {'min': 2.0, 'max': 15.0, 'type': 'float'}
            elif param.startswith("weight_"):
                if param == "weight_ma":
                    param_ranges[param] = {'min': 0.1, 'max': 0.5, 'type': 'float'}
                else:
                    param_ranges[param] = {'min': 0.0, 'max': 0.4, 'type': 'float'}
            elif param == "buy_threshold":
                param_ranges[param] = {'min': 0.2, 'max': 0.8, 'type': 'float'}
            elif param == "sell_threshold":
                param_ranges[param] = {'min': -0.8, 'max': -0.2, 'type': 'float'}
            else:
                logger.warning(f"パラメータ '{param}' の探索範囲が定義されていません")
        
        return param_ranges
    
    def _generate_random_params(self, param_ranges):
        """
        ランダムなパラメータ組み合わせを生成
        
        Parameters:
        -----------
        param_ranges : dict
            パラメータ名とその探索範囲のマッピング
            
        Returns:
        --------
        dict
            パラメータ名と値のマッピング
        """
        params_dict = {}
        
        for param, range_info in param_ranges.items():
            if range_info['type'] == 'int':
                params_dict[param] = np.random.randint(range_info['min'], range_info['max'] + 1)
            else:  # float
                params_dict[param] = np.random.uniform(range_info['min'], range_info['max'])
        
        return params_dict
    
    def _evaluate_params(self, params_dict, iteration):
        """
        パラメータ組み合わせを評価
        
        Parameters:
        -----------
        params_dict : dict
            パラメータ名と値のマッピング
        iteration : int
            イテレーション番号
            
        Returns:
        --------
        dict or None
            評価結果（メトリクスつき）
        """
        try:
            # パラメータを設定
            for param, value in params_dict.items():
                if hasattr(self.config, param):
                    setattr(self.config, param, value)
                else:
                    logger.warning(f"パラメータ '{param}' が見つかりません")
            
            # バックテストを実行
            metrics = self.backtester.run_backtest(debug_signals=False)
            
            # 結果をマージ
            result = {**params_dict, **metrics}
            
            # 重要なメトリクスだけを出力
            if 'profit_percent' in metrics:
                logger.debug(f"イテレーション {iteration}: 利益率 = {metrics['profit_percent']:.2f}%")
            
            return result
            
        except Exception as e:
            logger.error(f"パラメータ評価エラー（イテレーション {iteration}）: {e}")
            return None
    
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
    
    def _plot_optimization_results(self, results_df, params):
        """
        最適化結果を可視化
        
        Parameters:
        -----------
        results_df : pandas.DataFrame
            最適化結果
        params : list
            最適化したパラメータ名のリスト
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # パラメータグループ名を生成
            group_name = "_".join(params)
            
            # 利益率分布のヒストグラム
            plt.figure(figsize=(12, 8))
            plt.hist(results_df['profit_percent'], bins=20)
            plt.title(f"利益率分布 - {group_name}")
            plt.xlabel("利益率 (%)")
            plt.ylabel("頻度")
            plt.axvline(x=0, color='r', linestyle='--')
            plt.grid(True)
            plt.savefig(f"results/profit_distribution_{group_name}_{timestamp}.png")
            plt.close()
            
            # 2つ以上のパラメータがある場合はペアプロット
            if len(params) >= 2:
                plt.figure(figsize=(15, 15))
                scatter_params = params + ['profit_percent']
                sns.pairplot(results_df[scatter_params], kind='scatter', diag_kind='kde')
                plt.suptitle(f"パラメータ空間の散布図 - {group_name}")
                plt.savefig(f"results/param_pairplot_{group_name}_{timestamp}.png")
                plt.close()
            
            # ヒートマップ（パラメータ間の相関）
            plt.figure(figsize=(10, 8))
            corr = results_df[params + ['profit_percent']].corr()
            mask = np.triu(np.ones_like(corr, dtype=bool))
            sns.heatmap(
                corr, 
                mask=mask, 
                annot=True, 
                cmap='coolwarm',
                vmin=-1, 
                vmax=1
            )
            plt.title(f"パラメータ間の相関 - {group_name}")
            plt.tight_layout()
            plt.savefig(f"results/param_correlation_{group_name}_{timestamp}.png")
            plt.close()
            
            logger.info(f"最適化結果の可視化が完了しました: グループ {group_name}")
            
        except Exception as e:
            logger.error(f"結果の可視化エラー: {e}")
    
    def _save_optimization_results(self, result):
        """最適化結果を保存"""
        if not result:
            return
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # JSONとして保存
            with open(f"results/simple_optimization_{self.config.symbol}_{timestamp}.json", 'w') as f:
                json.dump(result, f, default=str, indent=4)
            
            logger.info(f"最適化結果を保存しました: results/simple_optimization_{self.config.symbol}_{timestamp}.json")
            
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
    
    def get_param_importance(self, num_iterations=30):
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
        
        # 探索範囲を定義
        param_ranges = self._create_param_ranges(all_params)
        
        # 既存の設定をバックアップ
        original_params = self._backup_config()
        
        try:
            # ランダムサンプリングの準備
            self.total_iterations = num_iterations
            self.completed_iterations = 0
            
            # 結果を格納する配列
            results = []
            
            # 並列処理でランダムサンプリング
            with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                # イテレーションごとにタスクを投入
                futures = []
                for i in range(num_iterations):
                    # ランダムなパラメータ組み合わせを生成
                    params_dict = self._generate_random_params(param_ranges)
                    # タスク投入
                    futures.append(executor.submit(self._evaluate_params, params_dict, i))
                
                # 結果を収集
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        if result:
                            results.append(result)
                            # 進捗報告
                            self.completed_iterations += 1
                            logger.info(f"重要度分析: {self.completed_iterations}/{num_iterations} 完了")
                    except Exception as e:
                        logger.error(f"評価タスクエラー: {e}")
            
            # 結果をDataFrameに変換
            if not results:
                logger.warning("重要度分析のための有効な結果がありません")
                return {}
                
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
        
        # 元の設定をバックアップ
        original_params = self._backup_config()
        
        try:
            # ランダムサンプリングの準備
            self.total_iterations = iterations
            self.completed_iterations = 0
            
            # 探索範囲を定義（重要なパラメータのみ）
            param_ranges = self._create_param_ranges(important_params)
            
            # 結果を格納する配列
            results = []
            
            # 並列処理でランダムサンプリング
            with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                # イテレーションごとにタスクを投入
                futures = []
                for i in range(iterations):
                    # ランダムなパラメータ組み合わせを生成
                    params_dict = self._generate_random_params(param_ranges)
                    # タスク投入
                    futures.append(executor.submit(self._evaluate_params, params_dict, i))
                
                # 結果を収集
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        if result:
                            results.append(result)
                            # 進捗報告
                            self.completed_iterations += 1
                            self._report_progress()
                    except Exception as e:
                        logger.error(f"評価タスクエラー: {e}")
            
            # 結果をDataFrameに変換
            if not results:
                logger.warning("最適化の有効な結果がありません")
                return {}
                
            results_df = pd.DataFrame(results)
            
            # 最良の結果を抽出
            best_result = results_df.loc[results_df['profit_percent'].idxmax()]
            
            # パラメータだけを抽出
            best_params = {param: best_result[param] for param in important_params if param in best_result}
            
            # 最終的な性能評価
            self._apply_params(best_params)
            final_metrics = self.backtester.run_backtest(debug_signals=False)
            
            # 結果をマージ
            result_dict = {**best_params, **final_metrics}
            
            # 結果を保存
            self._save_optimization_results(result_dict)
            
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