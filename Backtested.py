import warnings

warnings.filterwarnings('ignore')

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from datetime import datetime, timedelta
from itertools import product

import numpy as np
import pandas as pd
import ta
import yfinance as yf

np.random.seed(42)

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    'data': {
        'pairs': [
            'NZDUSD=X', 'AUDJPY=X', 'NZDJPY=X', 'GBPJPY=X', 
            'AUDUSD=X', 'EURGBP=X', 'EURUSD=X', 'USDJPY=X',
            'CADJPY=X', 'EURJPY=X', 'GBPAUD=X', 'EURAUD=X',
            'GBPUSD=X', 'USDCAD=X', 'CHFJPY=X', 'GBPCHF=X', 
            'EURCHF=X', 'AUDCHF=X', 'GBPCAD=X', 'AUDNZD=X', 
            'EURNZD=X', 'CADCHF=X', 'NZDCHF=X', 'NZDCAD=X',
        ],
        'years': 7,
        'test_days': 365,
    },
    'strategy': {
        'rsi_period': 14,
        'atr_period': 14,
    },
    'trading': {
        'stop_loss_pct': 1.5,
        'take_profit_pct': 6.0,
        'trailing_stop_trigger': 1.2,
        'trailing_stop_distance': 0.5,
        'max_hold_days': 10,
        'commission_pct': 0.02,
        'slippage_pct': 0.02,
        'breakeven_trigger': 0.5,
    },
    'risk': {
        'initial_capital': 1000,
        'risk_per_trade_pct': 32.0,     # ⭐ SWEET SPOT
        'max_positions': 12,
        'use_pyramiding': True,
        'max_pyramids': 3,
    },
    'adx_regime': {
        'use_adx_regime_filter': True,
        'adx_period': 14,
        'trending_threshold': 25,
        'transition_threshold': 18,
        'allow_trending': False,
        'allow_transition': True,
        'allow_ranging': False,
        'trending_size_mult': 0.5,
        # 'transition_size_mult': 1.8,    # ⭐ SWEET SPOT
        'transition_size_mult': 2.0,    # ⭐ NEWSWEET SPOT
        'ranging_size_mult': 0.0,
        'use_directional_bias': False,
        'require_bullish_bias': False,
        'use_confidence_score': False,
        'min_confidence': 0.0,
        'avoid_strengthening_trend': False,
    },
}

# ============================================================================
# DATA DOWNLOAD
# ============================================================================

def download_forex_data(pairs, years):
    """Download forex data from Yahoo Finance"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years * 365)
    data = {}
    
    print(f"Downloading {len(pairs)} pairs...")
    
    for pair in pairs:
        pair_name = pair.replace('=X', '')
        try:
            df = yf.download(pair, start=start_date, end=end_date, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.columns = [col.strip() for col in df.columns]
            if len(df) > 500:
                data[pair_name] = df
                print(f"  ✓ {pair_name}: {len(df)} bars")
            else:
                print(f"  ✗ {pair_name}: Only {len(df)} bars (skipped)")
        except Exception as e:
            print(f"  ✗ {pair_name}: {e}")
    
    return data


# ============================================================================
# ADX REGIME CLASSIFIER
# ============================================================================

def classify_adx_regime(df, config):
    """Classify market regime based on ADX"""
    df = df.copy()
    adx_config = config.get('adx_regime', {})
    
    if not adx_config.get('use_adx_regime_filter', False):
        df['regime'] = 'UNKNOWN'
        df['regime_bias'] = 'NEUTRAL'
        df['regime_confidence'] = 1.0
        df['regime_size_mult'] = 1.0
        df['regime_tradeable'] = True
        return df
    
    period = adx_config.get('adx_period', 14)
    trending_thresh = adx_config.get('trending_threshold', 25)
    transition_thresh = adx_config.get('transition_threshold', 20)
    
    # Calculate ADX
    adx_indicator = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close'], window=period)
    df['adx'] = adx_indicator.adx()
    df['plus_di'] = adx_indicator.adx_pos()
    df['minus_di'] = adx_indicator.adx_neg()
    
    # Classify regime
    def classify(adx):
        if pd.isna(adx):
            return 'UNKNOWN'
        if adx >= trending_thresh:
            return 'TRENDING'
        elif adx >= transition_thresh:
            return 'TRANSITION'
        else:
            return 'RANGING'
    
    df['regime'] = df['adx'].apply(classify)
    
    # Directional bias
    df['regime_bias'] = 'NEUTRAL'
    df.loc[df['plus_di'] > df['minus_di'], 'regime_bias'] = 'BULLISH'
    df.loc[df['minus_di'] > df['plus_di'], 'regime_bias'] = 'BEARISH'
    
    # Confidence and size multiplier
    df['regime_confidence'] = 0.7
    df['regime_size_mult'] = 1.0
    df['regime_tradeable'] = True
    
    return df


# ============================================================================
# SIGNAL GENERATION (FIXED - RELAXED CONDITIONS)
# ============================================================================

def add_enhanced_signals(df, config):
    """
    BALANCED: Good quality + More signals
    """
    df = df.copy()

    # ==================== TREND ====================
    df['ema5'] = ta.trend.ema_indicator(df['Close'], 5)
    df['ema8'] = ta.trend.ema_indicator(df['Close'], 8)
    df['ema13'] = ta.trend.ema_indicator(df['Close'], 13)
    df['ema21'] = ta.trend.ema_indicator(df['Close'], 21)
    df['ema50'] = ta.trend.ema_indicator(df['Close'], 50)

    # Trend conditions (relaxed slightly)
    df['ema_stack'] = (df['ema5'] > df['ema8']) & (df['ema8'] > df['ema21'])
    df['price_above_ema21'] = df['Close'] > df['ema21']
    df['ema21_rising'] = df['ema21'] > df['ema21'].shift(2)
    
    df['trend_ok'] = df['ema_stack'] | (df['price_above_ema21'] & df['ema21_rising'])

    # ==================== MOMENTUM ====================
    df['rsi'] = ta.momentum.rsi(df['Close'], 14)
    df['rsi_fast'] = ta.momentum.rsi(df['Close'], 7)
    
    # Wider RSI range
    df['momentum_ok'] = (df['rsi'] > 35) & (df['rsi'] < 70)
    
    # RSI rising or stable
    df['rsi_rising'] = df['rsi'] >= df['rsi'].shift(1) - 2

    # ==================== VOLATILITY ====================
    df['atr'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], 14)
    df['atr_sma'] = df['atr'].rolling(20).mean()
    df['volatility_ok'] = df['atr'] > df['atr_sma'] * 0.4

    # ==================== ADX ====================
    df = classify_adx_regime(df, config)
    df['adx_ok'] = df['adx'] > 18

    # ==================== MULTIPLE SETUPS ====================
    
    df['bullish_candle'] = (df['Close'] > df['Open'])
    df['strong_bullish'] = (df['Close'] - df['Open']) > (df['High'] - df['Low']) * 0.4
    
    # Setup 1: Pullback to any EMA
    df['pullback_ema8'] = (df['Low'] <= df['ema8'] * 1.003) & (df['Close'] > df['ema8']) & df['bullish_candle']
    df['pullback_ema13'] = (df['Low'] <= df['ema13'] * 1.004) & (df['Close'] > df['ema13']) & df['bullish_candle']
    df['pullback_ema21'] = (df['Low'] <= df['ema21'] * 1.005) & (df['Close'] > df['ema21']) & df['bullish_candle']
    
    # Setup 2: EMA crossovers
    df['ema_cross_5_8'] = (df['ema5'] > df['ema8']) & (df['ema5'].shift(1) <= df['ema8'].shift(1))
    df['ema_cross_8_21'] = (df['ema8'] > df['ema21']) & (df['ema8'].shift(1) <= df['ema21'].shift(1))
    
    # Setup 3: RSI bounce
    df['rsi_bounce'] = (df['rsi'] > 40) & (df['rsi'].shift(1) <= 40)
    
    # Setup 4: Strong bounce candle
    df['lower_wick'] = df[['Open', 'Close']].min(axis=1) - df['Low']
    df['body'] = abs(df['Close'] - df['Open'])
    df['hammer'] = (df['lower_wick'] > df['body'] * 1.2) & df['bullish_candle']
    
    # Setup 5: Breakout
    df['breakout'] = (df['Close'] > df['High'].shift(1)) & df['bullish_candle']
    
    # Setup 6: Higher high + higher low
    df['higher_low'] = df['Low'] > df['Low'].shift(1)
    df['higher_high'] = df['High'] > df['High'].shift(1)
    df['hh_hl'] = df['higher_low'] & df['higher_high'] & df['bullish_candle']
    
    # Combine setups
    df['setup_triggered'] = (
        df['pullback_ema8'] |
        df['pullback_ema13'] |
        df['pullback_ema21'] |
        df['ema_cross_5_8'] |
        df['ema_cross_8_21'] |
        df['rsi_bounce'] |
        df['hammer'] |
        df['breakout'] |
        df['hh_hl']
    )

    # ==================== FINAL ENTRY ====================
    df['entry_signal'] = (
        df['trend_ok'] &
        df['momentum_ok'] &
        df['adx_ok'] &
        df['setup_triggered']
    )

    return df

# ============================================================================
# TRADE CLASSES
# ============================================================================

class ForexTrade:
    """Base trade object"""
    
    def __init__(self, pair, entry_date, entry_price, direction, config, position_size):
        self.pair = pair
        self.entry_date = entry_date
        self.entry_price = entry_price * (1 + config['trading']['slippage_pct'] / 100)
        self.direction = direction
        self.days_held = 0
        self.config = config
        self.position_size = position_size
        self.pyramid_count = 0
        self.total_risk = position_size
        
        # Calculate stop and target
        stop_pct = config['trading']['stop_loss_pct'] / 100
        tp_pct = config['trading']['take_profit_pct'] / 100
        
        self.stop_distance = self.entry_price * stop_pct
        self.stop_price = self.entry_price * (1 - stop_pct)
        self.target_price = self.entry_price * (1 + tp_pct)
        
        self.breakeven_moved = False
        self.trailing_stop_active = False
        self.trailing_stop_price = None
        self.highest_price = entry_price
        
        self.regime = 'UNKNOWN'
        self.regime_confidence = 0.5
        self.regime_bias = 'NEUTRAL'
        self.atr_value = None
        self.partial_exit_done = False

    def try_pyramid(self, current_price, capital):
        """Add to position if in profit"""
        if not self.config['risk'].get('use_pyramiding', False):
            return 0
        
        max_pyramids = self.config['risk'].get('max_pyramids', 2)
        if self.pyramid_count >= max_pyramids:
            return 0
        
        # Only pyramid after +2R profit
        current_r = (current_price - self.entry_price) / self.stop_distance if self.stop_distance > 0 else 0
        if current_r < 2.0:
            return 0
        
        # Add 50% of original position
        pyramid_size = self.position_size * 0.5
        
        # Check limits
        if pyramid_size < capital * 0.01:
            return 0
        
        if (self.total_risk + pyramid_size) > (capital * 0.15):
            return 0
        
        self.pyramid_count += 1
        self.total_risk += pyramid_size
        return pyramid_size

    def check_exit(self, current_date, high, low, close, rsi):
        """Check if trade should exit"""
        self.days_held += 1
        
        # Track highest price
        if close > self.highest_price:
            self.highest_price = close
        
        # Get config values
        stop_pct = self.config['trading']['stop_loss_pct']
        tp_pct = self.config['trading']['take_profit_pct']
        be_trigger = self.config['trading']['breakeven_trigger']
        trail_trigger = self.config['trading']['trailing_stop_trigger']
        trail_distance = self.config['trading']['trailing_stop_distance']
        max_days = self.config['trading']['max_hold_days']
        
        # Calculate current profit %
        pct_profit = ((close - self.entry_price) / self.entry_price) * 100
        pct_high = ((high - self.entry_price) / self.entry_price) * 100
        pct_low = ((low - self.entry_price) / self.entry_price) * 100
        
        # 1. STOP LOSS
        if pct_low <= -stop_pct:
            exit_price = self.entry_price * (1 - stop_pct / 100)
            return True, exit_price, 'STOP_LOSS'
        
        # 2. TAKE PROFIT
        if pct_high >= tp_pct:
            exit_price = self.entry_price * (1 + tp_pct / 100)
            return True, exit_price, 'TAKE_PROFIT'
        
        # 3. BREAKEVEN
        if pct_profit >= be_trigger and not self.breakeven_moved:
            self.stop_price = self.entry_price * 1.001
            self.breakeven_moved = True
        
        # 4. TRAILING STOP
        if pct_profit >= trail_trigger:
            highest_pct = ((self.highest_price - self.entry_price) / self.entry_price) * 100
            trail_stop_pct = highest_pct - trail_distance
            
            if trail_stop_pct > 0:
                new_trail = self.entry_price * (1 + trail_stop_pct / 100)
                
                if not self.trailing_stop_active:
                    self.trailing_stop_active = True
                    self.trailing_stop_price = new_trail
                elif new_trail > self.trailing_stop_price:
                    self.trailing_stop_price = new_trail
        
        # Check trailing stop hit
        if self.trailing_stop_active and low <= self.trailing_stop_price:
            return True, self.trailing_stop_price, 'TRAILING_STOP'
        
        # Check breakeven stop hit
        if self.breakeven_moved and low <= self.stop_price:
            return True, self.stop_price, 'BREAKEVEN_EXIT'
        
        # 5. TIME EXIT
        if self.days_held >= max_days:
            if pct_profit > 0.5:
                return True, close, 'TIME_EXIT_PROFIT'
            elif pct_profit > -0.5:
                return True, close, 'TIME_EXIT_SCRATCH'
            else:
                return True, close, 'TIME_EXIT_LOSS'
        
        return False, None, None

class EnhancedForexTrade(ForexTrade):
    """
    FINAL: Relaxed early cut, let trades work longer
    """
    
    def __init__(self, pair, entry_date, entry_price, direction, config, position_size,
                 atr_value=None, regime='UNKNOWN', regime_confidence=0.5, regime_bias='NEUTRAL'):
        super().__init__(pair, entry_date, entry_price, direction, config, position_size)
        self.atr_value = atr_value
        self.regime = regime
        self.regime_confidence = regime_confidence
        self.regime_bias = regime_bias
        
        # Wide ATR-based stops
        if self.atr_value and self.atr_value > 0:
            self.stop_distance = self.atr_value * 3.0
            self.stop_price = self.entry_price - self.stop_distance
            self.target_price = self.entry_price + (self.stop_distance * 4.0)
        else:
            stop_pct = config['trading']['stop_loss_pct'] / 100
            self.stop_distance = self.entry_price * stop_pct
            self.stop_price = self.entry_price * (1 - stop_pct)
            self.target_price = self.entry_price * (1 + config['trading']['take_profit_pct'] / 100)
        
        self.profit_locked = False
        self.second_lock = False
    
    def check_exit(self, current_date, high, low, close, rsi):
        """FINAL: Relaxed early cut for better performance"""
        self.days_held += 1
        
        if close > self.highest_price:
            self.highest_price = close
        
        current_r = (close - self.entry_price) / self.stop_distance if self.stop_distance > 0 else 0
        
        # ========== EXIT LOGIC ==========
        
        # 1. STOP LOSS
        if low <= self.stop_price:
            return True, self.stop_price, 'STOP_LOSS'
        
        # 2. TAKE PROFIT
        if high >= self.target_price:
            return True, self.target_price, 'TAKE_PROFIT'
        
        # 3. FAST BREAKEVEN at +0.5R
        if current_r >= 0.5 and not self.breakeven_moved:
            self.stop_price = self.entry_price + (self.stop_distance * 0.1)
            self.breakeven_moved = True
        
        # 4. FIRST LOCK at +1R
        if current_r >= 1.0 and not self.profit_locked:
            self.stop_price = self.entry_price + (self.stop_distance * 0.4)
            self.profit_locked = True
        
        # 5. SECOND LOCK at +2R
        if current_r >= 2.0 and not self.second_lock:
            self.stop_price = self.entry_price + (self.stop_distance * 1.0)
            self.second_lock = True
        
        # 6. TRAILING after +2.5R
        if current_r >= 2.5:
            highest_r = (self.highest_price - self.entry_price) / self.stop_distance
            trail_r = highest_r - 0.5
            
            if trail_r > 0:
                new_trail = self.entry_price + (self.stop_distance * trail_r)
                
                if not self.trailing_stop_active:
                    self.trailing_stop_active = True
                    self.trailing_stop_price = new_trail
                elif new_trail > self.trailing_stop_price:
                    self.trailing_stop_price = new_trail
        
        if self.trailing_stop_active and low <= self.trailing_stop_price:
            return True, self.trailing_stop_price, 'TRAILING_STOP'
        
        if (self.breakeven_moved or self.profit_locked or self.second_lock) and low <= self.stop_price:
            return True, self.stop_price, 'PROFIT_PROTECTED'
        
        # 7. RSI OVERBOUGHT
        if rsi > 73 and current_r >= 1.5:
            return True, close, 'RSI_OVERBOUGHT'
        
        # 8. TIME EXITS - RELAXED early cut (only if very bad)
        max_days = self.config['trading']['max_hold_days']
        
        # Only cut early if VERY negative (was -0.3, now -0.5)
        if self.days_held >= max_days // 2:
            if current_r < -0.5:  # RELAXED from -0.3
                return True, close, 'TIME_CUT_EARLY'
        
        # Final time exit
        if self.days_held >= max_days:
            if current_r >= 1.0:
                return True, close, 'TIME_EXIT_PROFIT'
            elif current_r >= 0.2:  # RELAXED from 0.3
                return True, close, 'TIME_EXIT_SMALL'
            elif current_r >= -0.3:  # RELAXED from -0.2
                return True, close, 'TIME_EXIT_SCRATCH'
            else:
                return True, close, 'TIME_EXIT_LOSS'
        
        return False, None, None

# ============================================================================
# BACKTEST ENGINE
# ============================================================================

def run_backtest(signal_data, config):
    """
    EXTREME: 30%+ monthly target with high risk
    """
    
    test_days = config['data']['test_days']
    test_data = {pair: df.iloc[-test_days:].copy() for pair, df in signal_data.items()}
    all_dates = sorted(list(set().union(*[set(df.index) for df in test_data.values()])))
    
    capital = config['risk']['initial_capital']
    initial_capital = capital
    peak_capital = capital
    equity_curve = []
    open_trades = []
    closed_trades = []
    
    regime_trade_count = {'TRENDING': 0, 'TRANSITION': 0, 'RANGING': 0, 'UNKNOWN': 0}
    regime_wins = {'TRENDING': 0, 'TRANSITION': 0, 'RANGING': 0, 'UNKNOWN': 0}
    
    total_signals = 0
    total_trades = 0
    skipped_regime = 0
    consecutive_losses = 0
    consecutive_wins = 0
    
    monthly_returns = {}
    current_month = None
    month_start_capital = capital
    
    for date in all_dates:
        month_key = str(date)[:7] if hasattr(date, 'strftime') else str(date)[:7]
        if month_key != current_month:
            if current_month is not None:
                monthly_return = ((capital - month_start_capital) / month_start_capital) * 100
                monthly_returns[current_month] = monthly_return
            current_month = month_key
            month_start_capital = capital
        
        if capital > peak_capital:
            peak_capital = capital
        
        # ================================================================
        # MANAGE EXISTING TRADES
        # ================================================================
        for trade in open_trades[:]:
            pair = trade.pair
            if pair not in test_data or date not in test_data[pair].index:
                continue
            
            row = test_data[pair].loc[date]
            rsi = row.get('rsi', 50)
            
            # AGGRESSIVE PYRAMIDING
            if config['risk'].get('use_pyramiding', False):
                current_r = (row['Close'] - trade.entry_price) / trade.stop_distance if trade.stop_distance > 0 else 0
                
                # Pyramid earlier and more often
                if current_r > 1.2 and trade.pyramid_count < config['risk'].get('max_pyramids', 4):
                    pyramid_size = trade.position_size * 0.5  # 50% adds
                    if pyramid_size > capital * 0.02 and (trade.total_risk + pyramid_size) < capital * 0.50:
                        trade.pyramid_count += 1
                        trade.total_risk += pyramid_size
            
            should_exit, exit_price, exit_reason = trade.check_exit(
                date, row['High'], row['Low'], row['Close'], rsi
            )
            
            if should_exit:
                pct_move = ((exit_price - trade.entry_price) / trade.entry_price) * 100
                stop_loss_pct = config['trading']['stop_loss_pct']
                r_multiple = pct_move / stop_loss_pct
                
                r_multiple = max(min(r_multiple, 10.0), -1.0)  # Allow 10R wins
                
                pnl = trade.total_risk * r_multiple
                
                commission = trade.total_risk * (config['trading']['commission_pct'] / 100) * 2
                pnl -= commission
                
                max_win = capital * 0.50   # Allow 50% gains
                max_loss = capital * 0.10  # Cap 10% loss
                pnl = max(min(pnl, max_win), -max_loss)
                
                capital += pnl
                capital = max(capital, 50)
                
                is_win = pnl > 0
                
                if is_win:
                    consecutive_losses = 0
                    consecutive_wins += 1
                else:
                    consecutive_losses += 1
                    consecutive_wins = 0
                
                regime = trade.regime
                regime_trade_count[regime] = regime_trade_count.get(regime, 0) + 1
                if is_win:
                    regime_wins[regime] = regime_wins.get(regime, 0) + 1
                
                closed_trades.append({
                    'pair': pair,
                    'entry_date': trade.entry_date,
                    'exit_date': date,
                    'entry_price': trade.entry_price,
                    'exit_price': exit_price,
                    'pct_move': pct_move,
                    'r_multiple': r_multiple,
                    'pnl': pnl,
                    'pnl_pct': (pnl / trade.position_size * 100) if trade.position_size > 0 else 0,
                    'exit_reason': exit_reason,
                    'win': is_win,
                    'days_held': trade.days_held,
                    'pyramid_count': trade.pyramid_count,
                    'capital_after': capital,
                    'regime': regime,
                    'regime_confidence': trade.regime_confidence,
                    'regime_bias': trade.regime_bias,
                    'position_size': trade.position_size,
                    'total_risk': trade.total_risk,
                })
                
                open_trades.remove(trade)
        
        # ================================================================
        # SOFTER CIRCUIT BREAKERS (Let it run more)
        # ================================================================
        
        current_dd = ((capital - peak_capital) / peak_capital) * 100 if peak_capital > 0 else 0
        
        # Only reduce at extreme drawdowns
        if current_dd < -60:
            dd_mult = 0.5
        elif current_dd < -50:
            dd_mult = 0.7
        elif current_dd < -40:
            dd_mult = 0.85
        else:
            dd_mult = 1.0
        
        # Less aggressive loss reduction
        if consecutive_losses >= 5:
            loss_mult = 0.5
        elif consecutive_losses >= 3:
            loss_mult = 0.7
        elif consecutive_losses >= 2:
            loss_mult = 0.85
        else:
            loss_mult = 1.0
        
        # ================================================================
        # OPEN NEW TRADES - AGGRESSIVE
        # ================================================================
        max_positions = config['risk']['max_positions']
        
        # Even more positions on win streaks
        if consecutive_wins >= 5:
            max_positions = min(max_positions + 5, 20)
        elif consecutive_wins >= 4:
            max_positions = min(max_positions + 4, 18)
        elif consecutive_wins >= 3:
            max_positions = min(max_positions + 2, 16)
        elif consecutive_losses >= 4:
            max_positions = max(max_positions - 3, 5)
        
        if len(open_trades) < max_positions and capital > 50:
            for pair, df in test_data.items():
                if len(open_trades) >= max_positions:
                    break
                if any(t.pair == pair for t in open_trades):
                    continue
                if date not in df.index:
                    continue
                
                row = df.loc[date]
                
                if row.get('entry_signal', False):
                    total_signals += 1
                    
                    regime = row.get('regime', 'UNKNOWN')
                    regime_confidence = row.get('regime_confidence', 1.0)
                    regime_bias = row.get('regime_bias', 'NEUTRAL')
                    
                    adx_config = config.get('adx_regime', {})
                    
                    if regime == 'RANGING' and not adx_config.get('allow_ranging', True):
                        skipped_regime += 1
                        continue
                    if regime == 'TRANSITION' and not adx_config.get('allow_transition', True):
                        skipped_regime += 1
                        continue
                    if regime == 'TRENDING' and not adx_config.get('allow_trending', True):
                        skipped_regime += 1
                        continue
                    
                    # EXTREME POSITION SIZING
                    base_risk_pct = config['risk']['risk_per_trade_pct']
                    
                    # Bigger win streak bonuses
                    if consecutive_wins >= 5:
                        win_mult = 1.8
                    elif consecutive_wins >= 4:
                        win_mult = 1.6
                    elif consecutive_wins >= 3:
                        win_mult = 1.4
                    elif consecutive_wins >= 2:
                        win_mult = 1.2
                    else:
                        win_mult = 1.0
                    
                    if regime == 'TRENDING':
                        regime_mult = adx_config.get('trending_size_mult', 0.5)
                    elif regime == 'TRANSITION':
                        regime_mult = adx_config.get('transition_size_mult', 2.0)
                    else:
                        regime_mult = adx_config.get('ranging_size_mult', 0.5)
                    
                    # Less aggressive growth scaling
                    growth_ratio = capital / initial_capital
                    if growth_ratio > 20:
                        growth_mult = 0.4
                    elif growth_ratio > 10:
                        growth_mult = 0.6
                    elif growth_ratio > 5:
                        growth_mult = 0.75
                    else:
                        growth_mult = 1.0
                    
                    position_size = capital * (base_risk_pct / 100) * win_mult * loss_mult * dd_mult * regime_mult * growth_mult
                    
                    min_position = capital * 0.03
                    max_position = capital * 0.35  # Allow 35% per trade
                    position_size = max(min(position_size, max_position), min_position)
                    
                    total_open_risk = sum(t.total_risk for t in open_trades)
                    max_total_risk = capital * 0.90  # Allow 90% exposure
                    
                    if total_open_risk + position_size > max_total_risk:
                        available = max_total_risk - total_open_risk
                        if available < min_position:
                            continue
                        position_size = available
                    
                    atr_value = row.get('atr', None)
                    
                    trade = EnhancedForexTrade(
                        pair=pair,
                        entry_date=date,
                        entry_price=row['Close'],
                        direction=1,
                        config=config,
                        position_size=position_size,
                        atr_value=atr_value,
                        regime=regime,
                        regime_confidence=regime_confidence,
                        regime_bias=regime_bias
                    )
                    
                    open_trades.append(trade)
                    total_trades += 1
        
        equity_curve.append(capital)
    
    # Record final month
    if current_month is not None:
        monthly_return = ((capital - month_start_capital) / month_start_capital) * 100
        monthly_returns[current_month] = monthly_return
    
    # Close remaining trades
    for trade in open_trades[:]:
        pair = trade.pair
        if pair in test_data and len(test_data[pair]) > 0:
            last_row = test_data[pair].iloc[-1]
            last_date = test_data[pair].index[-1]
            exit_price = last_row['Close']
            
            pct_move = ((exit_price - trade.entry_price) / trade.entry_price) * 100
            r_multiple = pct_move / config['trading']['stop_loss_pct']
            r_multiple = max(min(r_multiple, 10.0), -1.0)
            
            pnl = trade.total_risk * r_multiple
            pnl -= trade.total_risk * (config['trading']['commission_pct'] / 100) * 2
            pnl = max(min(pnl, capital * 0.50), -capital * 0.10)
            
            capital += pnl
            is_win = pnl > 0
            
            regime = trade.regime
            regime_trade_count[regime] = regime_trade_count.get(regime, 0) + 1
            if is_win:
                regime_wins[regime] = regime_wins.get(regime, 0) + 1
            
            closed_trades.append({
                'pair': pair,
                'entry_date': trade.entry_date,
                'exit_date': last_date,
                'entry_price': trade.entry_price,
                'exit_price': exit_price,
                'pct_move': pct_move,
                'r_multiple': r_multiple,
                'pnl': pnl,
                'pnl_pct': (pnl / trade.position_size * 100) if trade.position_size > 0 else 0,
                'exit_reason': 'END_OF_TEST',
                'win': is_win,
                'days_held': trade.days_held,
                'pyramid_count': trade.pyramid_count,
                'capital_after': capital,
                'regime': regime,
                'regime_confidence': trade.regime_confidence,
                'regime_bias': trade.regime_bias,
                'position_size': trade.position_size,
                'total_risk': trade.total_risk,
            })
    
    # Print stats
    print(f"\n📊 Trade Execution Stats:")
    print(f"   Signals found: {total_signals}")
    print(f"   Skipped (regime): {skipped_regime}")
    print(f"   Trades opened: {total_trades}")
    print(f"   Trades closed: {len(closed_trades)}")
    
    if len(closed_trades) > 0:
        trades_df = pd.DataFrame(closed_trades)
        wins = trades_df[trades_df['win']]
        losses = trades_df[~trades_df['win']]
        
        if len(wins) > 0:
            print(f"   Avg Win: ${wins['pnl'].mean():.2f} ({wins['r_multiple'].mean():.2f}R)")
        if len(losses) > 0:
            print(f"   Avg Loss: ${losses['pnl'].mean():.2f} ({losses['r_multiple'].mean():.2f}R)")
        if len(wins) > 0 and len(losses) > 0 and losses['pnl'].sum() != 0:
            print(f"   Profit Factor: {abs(wins['pnl'].sum() / losses['pnl'].sum()):.2f}")
        
        print(f"\n   Exit Reasons:")
        exit_summary = trades_df.groupby('exit_reason').agg({
            'pnl': ['count', 'sum']
        }).round(2)
        exit_summary.columns = ['count', 'total_pnl']
        exit_summary = exit_summary.sort_values('count', ascending=False)
        
        for reason, row in exit_summary.iterrows():
            print(f"      {reason}: {int(row['count'])} trades, ${row['total_pnl']:+.2f}")
    
    # Monthly returns
    print(f"\n📅 Monthly Returns:")
    positive_months = 0
    target_months = 0
    extreme_months = 0
    
    for month, ret in sorted(monthly_returns.items()):
        if ret >= 50:
            status = "🔥"
            extreme_months += 1
            target_months += 1
            positive_months += 1
        elif ret >= 30:
            status = "🎯"
            target_months += 1
            positive_months += 1
        elif ret >= 25:
            status = "🎯"
            positive_months += 1
        elif ret >= 10:
            status = "✅"
            positive_months += 1
        elif ret > 0:
            status = "⚠️"
            positive_months += 1
        else:
            status = "❌"
        print(f"   {status} {month}: {ret:+.1f}%")
    
    avg_monthly = sum(monthly_returns.values()) / len(monthly_returns) if monthly_returns else 0
    print(f"\n   📊 Average Monthly: {avg_monthly:+.1f}%")
    print(f"   🔥 Months >= 50%: {extreme_months}/{len(monthly_returns)}")
    print(f"   🎯 Months >= 30%: {target_months}/{len(monthly_returns)}")
    print(f"   ✅ Positive Months: {positive_months}/{len(monthly_returns)}")
    
    # Max drawdown
    equity_array = np.array(equity_curve)
    running_max = np.maximum.accumulate(equity_array)
    drawdown = (equity_array - running_max) / running_max * 100
    max_dd = np.min(drawdown)
    print(f"   📉 Max Drawdown: {max_dd:.1f}%")
    
    if len(monthly_returns) > 1:
        monthly_vals = list(monthly_returns.values())
        avg_ret = np.mean(monthly_vals)
        std_ret = np.std(monthly_vals)
        if std_ret > 0:
            sharpe_monthly = avg_ret / std_ret
            print(f"   📈 Monthly Sharpe: {sharpe_monthly:.2f}")
    
    return closed_trades, equity_curve, all_dates, regime_trade_count, regime_wins

# ============================================================================
# MAIN EXECUTION
# ============================================================================

print("=" * 80)
print("🚀 FOREX BACKTESTER WITH ADX REGIME FILTER")
print("=" * 80)

# Download data
print("\n📥 Downloading forex data...")
all_data = download_forex_data(CONFIG['data']['pairs'], CONFIG['data']['years'])
print(f"\n✅ Loaded {len(all_data)} pairs")

if len(all_data) == 0:
    print("❌ No data loaded! Check internet connection.")
    exit()

# Generate signals
print("\n🔧 Generating signals...")
all_signals = {}
signal_counts = {}

for pair, df in all_data.items():
    try:
        df_filtered = add_enhanced_signals(df, CONFIG)
        all_signals[pair] = df_filtered
        
        test_data = df_filtered.iloc[-CONFIG['data']['test_days']:]
        signals = int(test_data['entry_signal'].sum()) if 'entry_signal' in test_data.columns else 0
        signal_counts[pair] = signals
        
        status = "✓" if signals > 0 else "⚠️"
        print(f"   {status} {pair}: {signals} signals")
    except Exception as e:
        print(f"   ❌ {pair}: {e}")
        signal_counts[pair] = 0

# Summary
total_signals = sum(signal_counts.values())
print(f"\n📊 Total signals: {total_signals}")

if total_signals == 0:
    print("\n⚠️ No signals generated! Checking conditions...")
    
    # Diagnostic
    first_pair = list(all_signals.keys())[0]
    test_data = all_signals[first_pair].iloc[-CONFIG['data']['test_days']:]
    
    conditions = ['trend_ok', 'trend_strong', 'momentum_ok', 'volatility_ok', 'adx_ok', 'setup_triggered']
    
    print(f"\n   Diagnostic for {first_pair}:")
    for cond in conditions:
        if cond in test_data.columns:
            count = int(test_data[cond].sum())
            pct = count / len(test_data) * 100
            status = "✅" if count > 0 else "❌"
            print(f"   {status} {cond}: {count} ({pct:.1f}%)")

# Run backtest
print("\n" + "=" * 80)
print("⏳ Running backtest...")
print("=" * 80)

all_trades, equity_curve, dates, regime_trade_count, regime_wins = run_backtest(all_signals, CONFIG)

# Results
print("\n" + "=" * 80)
print("📊 RESULTS")
print("=" * 80)

if len(all_trades) > 0:
    trades_df = pd.DataFrame(all_trades)
    
    wins = trades_df['win'].sum()
    losses = len(trades_df) - wins
    win_rate = (wins / len(trades_df)) * 100
    
    total_pnl = trades_df['pnl'].sum()
    final_capital = equity_curve[-1]
    initial_capital = CONFIG['risk']['initial_capital']
    total_return = ((final_capital - initial_capital) / initial_capital) * 100
    
    print(f"\n💰 Capital: ${initial_capital:,.0f} → ${final_capital:,.2f} ({total_return:+.1f}%)")
    print(f"📈 Trades: {len(trades_df)} ({wins}W / {losses}L)")
    print(f"🎯 Win Rate: {win_rate:.1f}%")
    print(f"💵 Total P/L: ${total_pnl:+,.2f}")
    
    # By regime
    print(f"\n📊 By Regime:")
    for regime in ['TRENDING', 'TRANSITION', 'RANGING']:
        count = regime_trade_count.get(regime, 0)
        wins_r = regime_wins.get(regime, 0)
        if count > 0:
            wr = (wins_r / count) * 100
            regime_pnl = trades_df[trades_df['regime'] == regime]['pnl'].sum()
            print(f"   {regime}: {count} trades, {wr:.1f}% WR, ${regime_pnl:+,.2f}")
    
    # Top pairs
    print(f"\n🌍 Top Pairs:")
    pair_pnl = trades_df.groupby('pair')['pnl'].sum().sort_values(ascending=False)
    for pair, pnl in pair_pnl.head(5).items():
        count = len(trades_df[trades_df['pair'] == pair])
        print(f"   {pair}: ${pnl:+,.2f} ({count} trades)")

else:
    print("\n❌ No trades executed!")
    print("   Check signal generation and entry conditions.")

print("\n" + "=" * 80)
print("✅ BACKTEST COMPLETE")
print("=" * 80)