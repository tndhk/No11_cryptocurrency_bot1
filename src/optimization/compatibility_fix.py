"""
scikit-optimize互換性修正モジュール

scikit-optimizeとNumPy/scikit-learnの新しいバージョン間の
互換性問題を解決するためのモンキーパッチを提供します。
"""

import numpy as np
import warnings
from skopt.space.space import Dimension, Integer
from sklearn.ensemble import ExtraTreesRegressor
import inspect

def apply_compatibility_patches():
    """互換性のあるモンキーパッチを適用します"""
    
    # 1. np.int問題を修正
    # 古いコードがnp.intを使用している場合、代わりにint（またはnp.int64）を使用するように修正
    try:
        from skopt.space.transformers import Transformer
        
        # Transformerクラスが存在するか確認
        if Transformer:
            # Integer Transformerのinverse_transform関数の修正
            old_inverse_transform = Transformer.inverse_transform
            
            def patched_inverse_transform(self, X):
                X_orig = old_inverse_transform(self, X)
                # isinstance(self, Identity)の代わりに、変換対象のdimensionによる判定
                if hasattr(self, '_dimension') and isinstance(self._dimension, Integer):
                    return np.round(X_orig).astype(int)  # np.intをintに変更
                return X_orig
            
            # モンキーパッチ適用
            Transformer.inverse_transform = patched_inverse_transform
    except (ImportError, AttributeError) as e:
        warnings.warn(f"Transformerクラスへのパッチ適用中にエラー: {e}. スキップします。")
    
    # 2. ExtraTreesRegressorのcriterion問題を修正
    def patched_forest_minimize(func, dimensions, base_estimator=None, **kwargs):
        from skopt.optimizer import forest_minimize
        
        # base_estimatorが指定されていない場合のみ修正
        if base_estimator is None:
            try:
                # ExtraTreesRegressorのパラメータを確認
                regressor_params = inspect.signature(ExtraTreesRegressor.__init__).parameters
                
                # 新しいバージョンのscikit-learnでは新しい基準を使用
                if 'criterion' in regressor_params:
                    # scikit-learn 1.0以降では'squared_error'、以前のバージョンでは'mse'
                    try:
                        base_estimator = ExtraTreesRegressor(n_estimators=100, criterion='squared_error')
                    except ValueError:
                        # 'squared_error'が失敗した場合は'mse'を試す
                        base_estimator = ExtraTreesRegressor(n_estimators=100, criterion='mse')
                else:
                    # デフォルト値を使用
                    base_estimator = ExtraTreesRegressor(n_estimators=100)
            except Exception as e:
                warnings.warn(f"ExtraTreesRegressorの設定中にエラー: {e}. デフォルト設定を使用します。")
                base_estimator = None  # デフォルトに戻す
        
        # 実際の forest_minimize を呼び出す
        return forest_minimize(func, dimensions, base_estimator=base_estimator, **kwargs)
    
    # 3. 変換器の問題を解決するためのパッチ
    try:
        from skopt.space.transformers import Transformer
        
        # Integer変換器の修正を試みる
        def safe_integer_transform(dimension, X):
            """Integerタイプの安全な変換"""
            X_round = np.round(X)  # 整数値に丸める
            
            # スカラーか配列かを判断
            if np.isscalar(X_round):
                return int(X_round)
            else:
                return X_round.astype(int)  # np.intではなくintを使用
        
        # 実際に利用可能な場合にのみパッチを適用
        if hasattr(Integer, 'transform'):
            old_integer_transform = Integer.transform
            
            def patched_integer_transform(self, X):
                try:
                    return old_integer_transform(self, X)
                except Exception:
                    # 元のメソッドが失敗した場合の代替実装
                    return safe_integer_transform(self, X)
            
            Integer.transform = patched_integer_transform
    except Exception as e:
        warnings.warn(f"Integer変換器へのパッチ適用中にエラー: {e}. スキップします。")
    
    # 警告表示
    warnings.warn(
        "互換性パッチを適用しました。scikit-optimize、scikit-learn、NumPyの"
        "バージョンに互換性の問題がある可能性があります。"
        "最新バージョンにアップデートすることを検討してください。"
    )
    
    return patched_forest_minimize