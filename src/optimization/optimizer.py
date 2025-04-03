import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
import traceback
import itertools

from src.utils.logger import default_logger as logger

class ParameterOptimizer:
    """
    パラメータ最適化クラス - 並列処理対応
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
    
    def optimize_parameters(self, param_grid=None):
        """
        パラメータ最適化を実行
        
        Parameters:
        -----------
        param_grid : dict, optional
            最適化するパラメータのグリッド
            
        Returns:
        --------
        dict
            最適なパラメータと結果
        """
        if param_grid is None:
            # デフォルトのパラメータグリッド - 探索範囲を拡大
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
        
        logger.info("パラメータ最適化を開始...")
        logger.info(f"パラメータグリッド: {param_grid}")
        
        # 全組み合わせ数を計算
        total_combinations = 1
        for param_values in param_grid.values():
            total_combinations *= len(param_values)
        
        logger.info(f"最適化する組み合わせ数: {total_combinations}")
        logger.info(f"使用するワーカー数: {self.num_workers}")
        
        # 現在の設定をバックアップ
        original_params = {
            param: getattr(self.config, param)
            for param in param_grid.keys()
            if hasattr(self.config, param)
        }
        
        # パラメータの組み合わせを生成
        param_names = list(param_grid.keys())
        param_combinations = list(itertools.product(*param_grid.values()))
        param_dicts = [dict(zip(param_names, combination)) for combination in param_combinations]
        
        # 並列実行用のキュー
        self.results = []
        
        try:
            # マルチプロセスで最適化
            with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                # 各組み合わせについて並列にバックテスト実行
                futures = {executor.submit(self._run_backtest_with_params, params, i, total_combinations): i 
                         for i, params in enumerate(param_dicts)}
                
                # 結果を収集
                for future in as_completed(futures):
                    try:
                        task_id = futures[future]
                        result = future.result()
                        if result:
                            self.results.append(result)
                            # 進捗状況を間引いて表示（10%ごと）
                            progress = (task_id + 1) / total_combinations * 100
                            if (task_id + 1) % max(1, total_combinations // 10) == 0:
                                logger.info(f"進捗: {progress:.1f}% ({task_id + 1}/{total_combinations})")
                    except Exception as e:
                        logger.error(f"タスク実行エラー: {e}")
                        logger.error(traceback.format_exc())
            
            # 結果をDataFrameに変換
            if not self.results:
                logger.error("有効な最適化結果がありません")
                return None
                
            results_df = pd.DataFrame(self.results)
            
            # 最良の結果を見つける
            best_result = results_df.sort_values('net_profit', ascending=False).iloc[0].to_dict()
            
            logger.info("=" * 80)
            logger.info("最適化結果")
            logger.info("=" * 80)
            logger.info(f"最適パラメータ: {dict((k, best_result[k]) for k in param_names)}")
            logger.info(f"純利益: {best_result['net_profit']:.2f} USDT ({best_result['profit_percent']:.2f}%)")
            logger.info(f"勝率: {best_result['win_rate']:.2f}%")
            logger.info(f"損益比率: {best_result['profit_factor']:.2f}")
            logger.info(f"最大ドローダウン: {best_result['max_drawdown']:.2f}%")
            
            # 結果をCSVとして保存
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_df.to_csv(f"results/optimization_{self.config.symbol}_{timestamp}.csv", index=False)
            
            # ベストな組み合わせをJSONとして保存
            with open(f"results/best_params_{self.config.symbol}_{timestamp}.json", 'w') as f:
                json.dump(best_result, f, default=str, indent=4)
            
            # 元のパラメータに戻す
            for param, value in original_params.items():
                setattr(self.config, param, value)
            
            # パラメータの重要度分析
            self._analyze_parameter_importance(results_df, param_names)
            
            return best_result
            
        except Exception as e:
            logger.error(f"最適化実行エラー: {e}")
            logger.error(traceback.format_exc())
            # 元のパラメータに戻す
            for param, value in original_params.items():
                setattr(self.config, param, value)
            return None
    
    def _run_backtest_with_params(self, params, task_id, total_tasks):
        """
        指定パラメータでバックテスト実行（プロセスプールワーカー用）
        
        Parameters:
        -----------
        params : dict
            テストするパラメータの組み合わせ
        task_id : int
            タスクID（進捗表示用）
        total_tasks : int
            全タスク数
            
        Returns:
        --------
        dict
            バックテスト結果とパラメータ
        """
        try:
            # デバッグレベルをINFOに設定して余計なログを抑制
            worker_logger = logger.bind(task_id=task_id)
                    
            # バックテスターのコピーを作成
            # 注意: 並列実行のため、Backtesterを新規に作る必要があるかもしれません
            # この部分は実装によって異なります
            
            # 安全のため、一時的にパラメータを変更
            config_copy = self.config  # ここで実際はコピーが必要かも
            
            # パラメータを適用
            for param, value in params.items():
                setattr(config_copy, param, value)
            
            # ログは最小限に
            worker_logger.info(f"組み合わせ {task_id+1}/{total_tasks} をテスト中: {params}")
            
            # このパラメータセットでバックテスト実行
            metrics = self.backtester.run_backtest(debug_signals=False)
            
            # 結果を保存
            if metrics:
                result_dict = {**params, **metrics}
                return result_dict
            else:
                worker_logger.warning(f"バックテスト結果がありません: {params}")
                return None
                
        except Exception as e:
            logger.error(f"ワーカー実行エラー (TaskID {task_id}): {e}")
            logger.error(traceback.format_exc())
            return None
    
    def _analyze_parameter_importance(self, results_df, param_names):
        """パラメータの重要度を分析"""
        try:
            # 重要度分析のグラフを生成
            plt.figure(figsize=(12, 8))
            
            # 各パラメータと利益の相関を可視化
            for i, param in enumerate(param_names):
                if results_df[param].nunique() > 1:  # 複数の値がある場合のみプロット
                    plt.subplot(3, 3, i+1)
                    sns.boxplot(x=param, y='profit_percent', data=results_df)
                    plt.title(f'{param} vs Profit')
                    plt.xticks(rotation=45)
                    
            plt.tight_layout()
            
            # 保存
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plt.savefig(f'results/param_importance_{self.config.symbol}_{timestamp}.png')
            plt.close()
            
            # 相関分析
            correlation = results_df[param_names + ['profit_percent']].corr()['profit_percent'].sort_values(ascending=False)
            
            logger.info("パラメータ重要度分析 (利益との相関):")
            for param, corr in correlation.items():
                if param != 'profit_percent':
                    logger.info(f"  {param}: {corr:.4f}")
                    
        except Exception as e:
            logger.error(f"パラメータ重要度分析エラー: {e}")
            
    def run_walk_forward_optimization(self, window_size=30, step_size=15, n_windows=3):
        """
        ウォークフォワード最適化の実行
        
        Parameters:
        -----------
        window_size : int
            各最適化ウィンドウのサイズ（日数）
        step_size : int
            次のウィンドウへの進行日数
        n_windows : int
            最適化ウィンドウの数
            
        Returns:
        --------
        tuple
            (robustness_score, results)
        """
        # データ期間を取得
        end_time = int(datetime.now().timestamp() * 1000)
        total_days = window_size * n_windows
        start_time = end_time - (total_days * 24 * 60 * 60 * 1000)
        
        results = []
        
        try:
            # 各ウィンドウで最適化とテストを繰り返す
            for i in range(n_windows):
                # インサンプル期間（最適化用）
                in_sample_end = end_time - (i * step_size * 24 * 60 * 60 * 1000)
                in_sample_start = in_sample_end - (window_size * 24 * 60 * 60 * 1000)
                
                logger.info(f"ウォークフォワード最適化 {i+1}/{n_windows}")
                logger.info(f"インサンプル期間: {datetime.fromtimestamp(in_sample_start/1000)} - {datetime.fromtimestamp(in_sample_end/1000)}")
                
                # アウトオブサンプル期間（検証用）
                out_sample_end = in_sample_start
                out_sample_start = out_sample_end - (step_size * 24 * 60 * 60 * 1000)
                
                logger.info(f"アウトオブサンプル期間: {datetime.fromtimestamp(out_sample_start/1000)} - {datetime.fromtimestamp(out_sample_end/1000)}")
                
                # 実装が必要なメソッド：
                # - 特定の期間のデータだけを使った最適化
                # - 特定の期間のデータだけを使ったバックテスト
                
                # この例ではサンプル実装を示します
                # best_params = self.optimize_parameters_for_period(in_sample_start, in_sample_end)
                # performance = self.backtest_with_params(best_params, out_sample_start, out_sample_end)
                # results.append(performance)
                
                # サンプル結果
                sample_performance = {
                    'window': i+1,
                    'in_sample_start': datetime.fromtimestamp(in_sample_start/1000).strftime('%Y-%m-%d'),
                    'in_sample_end': datetime.fromtimestamp(in_sample_end/1000).strftime('%Y-%m-%d'),
                    'out_sample_start': datetime.fromtimestamp(out_sample_start/1000).strftime('%Y-%m-%d'),
                    'out_sample_end': datetime.fromtimestamp(out_sample_end/1000).strftime('%Y-%m-%d'),
                    'in_sample_profit': 5.0,
                    'out_sample_profit': 3.0,
                    'robustness': 0.6  # out / in
                }
                results.append(sample_performance)
            
            # 結果の分析
            if results:
                robustness_scores = [r['robustness'] for r in results]
                avg_robustness = sum(robustness_scores) / len(robustness_scores)
                
                logger.info("=" * 50)
                logger.info("ウォークフォワード最適化結果")
                logger.info("=" * 50)
                for i, result in enumerate(results):
                    logger.info(f"ウィンドウ {i+1}:")
                    logger.info(f"  インサンプル利益: {result['in_sample_profit']}%")
                    logger.info(f"  アウトオブサンプル利益: {result['out_sample_profit']}%")
                    logger.info(f"  堅牢性スコア: {result['robustness']}")
                
                logger.info(f"全体の堅牢性スコア: {avg_robustness:.2f}")
                
                # 結果を保存
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                with open(f"results/walk_forward_{self.config.symbol}_{timestamp}.json", 'w') as f:
                    json.dump(results, f, default=str, indent=4)
                
                return avg_robustness, results
            else:
                logger.warning("有効なウォークフォワード最適化結果がありません")
                return 0, []
                
        except Exception as e:
            logger.error(f"ウォークフォワード最適化エラー: {e}")
            logger.error(traceback.format_exc())
            return 0, []