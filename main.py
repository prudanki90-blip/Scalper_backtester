import os
import time
import urllib.request
import json
import duckdb
import pandas as pd

# ==========================================
# 1. CORE ENGINE & NATIVE INDICATORS
# ==========================================
class MobileTerminalApp:
    def __init__(self, db_path="app_warehouse.db"):
        self.conn = duckdb.connect(db_path)
        self.create_tables()
        
    def create_tables(self):
        """Creates storage using your active DuckDB database engine."""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS market_data (
                timestamp TIMESTAMP PRIMARY KEY,
                open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE, volume DOUBLE,
                rsi DOUBLE, bb_lower DOUBLE, bb_middle DOUBLE, bb_upper DOUBLE
            )
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS trade_logs (
                id INTEGER PRIMARY KEY DEFAULT nextval('trade_id_seq'),
                direction VARCHAR, lot_size DOUBLE, entry_price DOUBLE, 
                exit_price DOUBLE, pnl DOUBLE, reason VARCHAR, status VARCHAR
            )
        """)
        try:
            self.conn.execute("CREATE SEQUENCE trade_id_seq;")
        except:
            pass

    def calculate_native_indicators(self, df):
        """Pure math processing layer using native Pandas analytics."""
        # Bollinger Bands (20 Period, 2 Standard Deviations)
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        rolling_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (rolling_std * 2)
        df['bb_lower'] = df['bb_middle'] - (rolling_std * 2)

        # RSI Momentum (14 Period)
        change = df['close'].diff()
        gain = change.mask(change < 0, 0)
        loss = -change.mask(change > 0, 0)
        avg_gain = gain.ewm(com=13, adjust=False).mean()
        avg_loss = loss.ewm(com=13, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, 0.00001)
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Fill placeholders safely
        df.fillna(method='bfill', inplace=True)
        return df

    def one_click_download(self, pair="BTCUSDT"):
        """Downloads historical charts via built-in system network modules."""
        print(f"🌐 Fetching {pair} 1-minute historical nodes...")
        url = f"https://binance.com{pair}&interval=1m&limit=150"
        
        # Pull data natively without requiring the 'requests' library
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            raw_json = json.loads(response.read().decode())
            
        data = []
        for row in raw_json:
            data.append({
                'timestamp': pd.to_datetime(row[0], unit='ms'),
                'open': float(row[1]), 'high': float(row[2]),
                'low': float(row[3]), 'close': float(row[4]), 'volume': float(row[5])
            })
            
        df = pd.DataFrame(data)
        df = self.calculate_native_indicators(df)
        
        # Save straight to local embedded DuckDB warehouse
        self.conn.register('temp_df', df)
        self.conn.execute("INSERT OR REPLACE INTO market_data SELECT * FROM temp_df")
        print("💾 Storage finalized. Data locked into database successfully.\n")

    def run_animated_backtest(self, simulation_speed=0.2):
        """Runs the loop, drawing moving candles and trading stats in real-time."""
        candles = self.conn.execute("SELECT * FROM market_data ORDER BY timestamp ASC").df()
        self.conn.execute("DELETE FROM trade_logs")
        
        contract_size = 100000
        spread_friction = 1.0 * 0.0001
        
        print("▶️ Booting Automated Scalper Core...")
        time.sleep(1)

        for i in range(len(candles)):
            candle = candles.iloc[i].to_dict()
            
            # 1. Evaluate Open Trades
            open_trades = self.conn.execute("SELECT * FROM trade_logs WHERE status='OPEN'").df()
            if not open_trades.empty:
                for idx, pos in open_trades.iterrows():
                    if candle['high'] >= (pos['entry_price'] + 0.0015):
                        pnl = 0.0015 * pos['lot_size'] * contract_size
                        self.conn.execute("UPDATE trade_logs SET status='CLOSED', exit_price=?, pnl=? WHERE id=?", 
                                          (pos['entry_price'] + 0.0015, pnl, pos['id']))
                    elif candle['low'] <= (pos['entry_price'] - 0.0010):
                        pnl = -0.0010 * pos['lot_size'] * contract_size
                        self.conn.execute("UPDATE trade_logs SET status='CLOSED', exit_price=?, pnl=? WHERE id=?", 
                                          (pos['entry_price'] - 0.0010, pnl, pos['id']))

            # 2. Strategy Rules (Mean Reversion Trigger)
            has_active = self.conn.execute("SELECT COUNT(*) FROM trade_logs WHERE status='OPEN'").fetchone()[0]
            if has_active == 0:
                if candle['close'] <= candle['bb_lower'] and candle['rsi'] <= 35:
                    reason = f"Price (${candle['close']:.2f}) broke Bollinger Floor (${candle['bb_lower']:.2f}) at RSI ({candle['rsi']:.1f})."
                    fill = candle['open'] + spread_friction
                    self.conn.execute("""
                        INSERT INTO trade_logs (direction, lot_size, entry_price, status, reason, pnl) 
                        VALUES ('BUY', 0.1, ?, 'OPEN', ?, 0.0)
                    """, (fill, reason))

            # 3. RENDER THE LIVE DASHBOARD ON YOUR PHONE SCREEN
            # Clear terminal window for crisp frame refresh animation
            os.system('clear' if os.name == 'posix' else 'cls')
            
            # Fetch updated metrics
            closed_trades = self.conn.execute("SELECT * FROM trade_logs WHERE status='CLOSED'").df()
            total_t = len(closed_trades)
            wins = len(closed_trades[closed_trades['pnl'] > 0]) if total_t > 0 else 0
            win_rate = (wins / total_t * 100) if total_t > 0 else 0.0
            net_pnl = closed_trades['pnl'].sum() if total_t > 0 else 0.0
            
            # Output Screen Panel
            print("=============================================================")
            print(f"📊 LIVE MOVING CANDLES GRAPH TERMINAL  |  SPEED: {simulation_speed}s")
            print("=============================================================")
            print(f" TIME:  {candle['timestamp']}")
            print(f" PRICE: ${candle['close']:.2f} | BB-MID: ${candle['bb_middle']:.2f}")
            print(f" RSI:   [{'#' * int(candle['rsi']/5)}{'-' * (20 - int(candle['rsi']/5))}] ({candle['rsi']:.1f})")
            print("=============================================================")
            print("📈 TRADER STATISTICS TABLE")
            print("-------------------------------------------------------------")
            print(f" Total Trades: {total_t:<6} | Win Rate: {win_rate:.1f}% | Net Profit: ${net_pnl:.2f}")
            print("=============================================================")
            print("📜 ACTIVE RECENT TRADE EXPLANATIONS LOGS")
            print("-------------------------------------------------------------")
            
            all_trades = self.conn.execute("SELECT direction, pnl, reason FROM trade_logs ORDER BY id DESC LIMIT 3").fetchall()
            for t in all_trades:
                print(f" -> [{t[0]}] PnL: ${t[1]:.2f} | Reason: {t[2]}")
            print("=============================================================")
            
            # Controls animation cadence frame rate
            time.sleep(simulation_speed)

# ==========================================
# 4. ONE-CLICK INITIALIZATION RUNNER
# ==========================================
if __name__ == "__main__":
    app = MobileTerminalApp()
    
    # Action 1: Automated download & store data framework execution
    app.one_click_download("BTCUSDT")
    
    # Action 2: Change this number to control how fast the simulation ticks (e.g., 0.05 is fast, 0.5 is slow)
    USER_SPEED_SETTING = 0.1 
    
    # Action 3: Boot the active engine loop mapping dashboard onto your screen
    app.run_animated_backtest(simulation_speed=USER_SPEED_SETTING)
      
