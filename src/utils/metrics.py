import pandas as pd
import numpy as np

def calculate_performance_metrics(trades, balance_history, initial_balance):
    """
    トレードのパフォーマンスメトリクスを計算
    
    Parameters:
    -----------
    trades: list
        トレード履歴のリスト
    balance_history: list
        残高履歴のリスト (timestamp, balance)
    initial_balance: float
        初期残高
        
    Returns:
    --------
    dict
        パフォーマンスメトリクス
    """
    if not trades:
        return {
            'net_profit': 0,
            'profit_percent': 0,
            'total_trades': 0,
            'closed_trades': 0,
            'winning_trades': 0,
            'win_rate': 0,
            'profit_factor': 0,
            'max_drawdown': 0,
            'avg_profit': 0,
            'avg_loss': 0,
            'avg_hold_time': 0
        }
    
    # 取引統計
    buy_trades = [t for t in trades if t['type'] == 'BUY']
    sell_trades = [t for t in trades if t['type'] == 'SELL']
    
    total_trades = len(buy_trades)
    closed_trades = len(sell_trades)
    
    # 最終残高
    final_balance = balance_history[-1][1] if balance_history else initial_balance
    
    # 利益計算
    net_profit = final_balance - initial_balance
    profit_percent = (final_balance / initial_balance - 1) * 100 if initial_balance > 0 else 0
    
    # 勝敗統計
    winning_trades = [t for t in sell_trades if t.get('net_profit', 0) > 0]
    losing_trades = [t for t in sell_trades if t.get('net_profit', 0) <= 0]
    
    win_rate = (len(winning_trades) / closed_trades * 100) if closed_trades > 0 else 0
    
    # 損益比率
    total_profit = sum([t.get('net_profit', 0) for t in winning_trades]) if winning_trades else 0
    total_loss = sum([t.get('net_profit', 0) for t in losing_trades]) if losing_trades else 0
    profit_factor = abs(total_profit / total_loss) if total_loss != 0 else float('inf')
    
    # 平均利益/損失
    avg_profit = total_profit / len(winning_trades) if winning_trades else 0
    avg_loss = total_loss / len(losing_trades) if losing_trades else 0
    
    # 保有時間
    hold_times = [t.get('hold_duration', 0) for t in sell_trades]
    avg_hold_time = sum(hold_times) / len(hold_times) if hold_times else 0
    
    # 最大ドローダウン
    balance_series = pd.DataFrame(balance_history, columns=['timestamp', 'balance'])['balance'] if balance_history else pd.Series([initial_balance])
    rolling_max = balance_series.cummax()
    drawdown = (rolling_max - balance_series) / rolling_max * 100
    max_drawdown = drawdown.max() if not drawdown.empty else 0
    
    return {
        'net_profit': net_profit,
        'profit_percent': profit_percent,
        'total_trades': total_trades,
        'closed_trades': closed_trades, 
        'winning_trades': len(winning_trades),
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'max_drawdown': max_drawdown,
        'avg_profit': avg_profit,
        'avg_loss': avg_loss,
        'avg_hold_time': avg_hold_time
    }

def calculate_drawdown(balance_history):
    """
    ドローダウンを計算
    
    Parameters:
    -----------
    balance_history: list
        残高履歴のリスト (timestamp, balance)
        
    Returns:
    --------
    float
        現在のドローダウン(%)
    """
    if not balance_history:
        return 0.0
    
    # 履歴からの残高推移
    current_balance = balance_history[-1][1]
    
    # これまでの残高のピーク
    peak_balance = max([b[1] for b in balance_history])
    
    if peak_balance == 0:
        return 0.0
    
    drawdown_percent = (peak_balance - current_balance) / peak_balance * 100
    return drawdown_percent