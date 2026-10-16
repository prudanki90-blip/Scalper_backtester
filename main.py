import os
import time
import requests
import duckdb
import pandas as pd
from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ==========================================
# 1. INITIALIZE FASTAPI & WEB SYSTEM BRIDGE
# ==========================================
app = FastAPI(title="Unified Scalper Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# 2. CORE BACKEND ARCHITECTURE MACHINERY
# ==========================================
class UnifiedScalperEngine:
    def __init__(self, db_path="app_warehouse.db"):
        self.conn = duckdb.connect(db_path)
        self.simulation_speed = 0.05 
        self.is_running = False
        self.create_tables()
        
    def create_tables(self):
        """Creates time-series storage and execution tracking tables inside DuckDB."""
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
        """
        Pure math engine. Calculates RSI and Bollinger Bands directly 
        using native pandas functions to prevent server build crashes.
        """
        # Bollinger Bands Math (20 Period, 2 Standard Deviations)
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        rolling_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (rolling_std * 2)
        df['bb_lower'] = df['bb_middle'] - (rolling_std * 2)

        # RSI Math (14 Period Momentum)
        change = df['close'].diff()
        gain = change.mask(change < 0, 0)
        loss = -change.mask(change > 0, 0)
        
        avg_gain = gain.ewm(com=13, adjust=False).mean()
        avg_loss = loss.ewm(com=13, adjust=False).mean()
        
        rs = avg_gain / avg_loss.replace(0, 0.00001)
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Fill placeholder rows safely
        df['rsi'] = df['rsi'].fillna(50)
        df['bb_middle'] = df['bb_middle'].fillna(df['close'])
        df['bb_upper'] = df['bb_upper'].fillna(df['close'])
        df['bb_lower'] = df['bb_lower'].fillna(df['close'])
        
        return df

    def download_and_store_data(self, pair: str):
        """One-Click Downloader: Pulls raw data, calculates indicators, and loads database."""
        url = f"https://binance.com{pair}&interval=1m&limit=500"
        response = requests.get(url).json()
        
        data = []
        for row in response:
            data.append({
                'timestamp': pd.to_datetime(row[0], unit='ms'),
                'open': float(row[1]), 'high': float(row[2]),
                'low': float(row[3]), 'close': float(row[4]), 'volume': float(row[5])
            })
        
        df = pd.DataFrame(data)
        df = self.calculate_native_indicators(df)
        
        # Save straight to local embedded DuckDB database file
        self.conn.register('temp_df', df)
        self.conn.execute("INSERT OR REPLACE INTO market_data SELECT * FROM temp_df")
        print(f"📊 {pair} data and metrics stored successfully using native engines.")

    def run_simulation_loop(self, speed: float, risk_pct: float, balance: float):
        """Bar-by-Bar Scalper execution engine loop."""
        self.simulation_speed = speed
        self.is_running = True
        
        candles = self.conn.execute("SELECT * FROM market_data ORDER BY timestamp ASC").df()
        contract_size = 100000
        spread_slippage_friction = 1.5 * 0.0001 
        
        self.conn.execute("DELETE FROM trade_logs")

        for i in range(len(candles)):
            if not self.is_running:
                break
                
            candle = candles.iloc[i].to_dict()
            
            # A. Evaluate active open positions first
            open_positions = self.conn.execute("SELECT * FROM trade_logs WHERE status='OPEN'").df()
            if not open_positions.empty:
                for idx, pos in open_positions.iterrows():
                    if candle['high'] >= (pos['entry_price'] + 0.0015): 
                        pnl = 0.0015 * pos['lot_size'] * contract_size
                        self.conn.execute("UPDATE trade_logs SET status='CLOSED', exit_price=?, pnl=? WHERE id=?", 
                                          (pos['entry_price'] + 0.0015, pnl, pos['id']))
                    elif candle['low'] <= (pos['entry_price'] - 0.0010): 
                        pnl = -0.0010 * pos['lot_size'] * contract_size
                        self.conn.execute("UPDATE trade_logs SET status='CLOSED', exit_price=?, pnl=? WHERE id=?", 
                                          (pos['entry_price'] - 0.0010, pnl, pos['id']))

            # B. Strategy evaluation entry checks
            has_active_trade = self.conn.execute("SELECT COUNT(*) FROM trade_logs WHERE status='OPEN'").fetchone()[0]
            if has_active_trade == 0:
                if candle['close'] <= candle['bb_lower'] and candle['rsi'] <= 35: 
                    reason_desc = f"Price (${candle['close']:.2f}) pierced lower band (${candle['bb_lower']:.2f}) at oversold RSI ({candle['rsi']:.1f}). Buying reversion."
                    fill_price = candle['open'] + spread_slippage_friction
                    
                    self.conn.execute("""
                        INSERT INTO trade_logs (direction, lot_size, entry_price, status, reason, pnl) 
                        VALUES ('BUY', 0.1, ?, 'OPEN', ?, 0.0)
                    """, (fill_price, reason_desc))

            time.sleep(self.simulation_speed)
            
        self.is_running = False

backend_system = UnifiedScalperEngine()

# ==========================================
# 3. INTERACTIVE WEB ENDPOINT ROUTES (APIs)
# ==========================================
class ConfigSchema(BaseModel):
    pair: str
    speed: float
    risk: float
    funds: float

@app.post("/api/download")
def download(pair: str):
    backend_system.download_and_store_data(pair)
    return {"message": "Downloaded"}

@app.post("/api/start")
def start(config: ConfigSchema, tasks: BackgroundTasks):
    if not backend_system.is_running:
        tasks.add_task(backend_system.run_simulation_loop, speed=config.speed, risk=config.risk, balance=config.funds)
        return {"status": "running"}
    return {"status": "already_running"}

@app.get("/api/dashboard")
def dashboard():
    trades = backend_system.conn.execute("SELECT * FROM trade_logs ORDER BY id DESC").df().to_dict(orient="records")
    closed = [t for t in trades if t['status'] == 'CLOSED']
    
    total = len(closed)
    wins = len([t for t in closed if t['pnl'] > 0])
    win_rate = (wins / total * 100) if total > 0 else 0.0
    net_pnl = sum([t['pnl'] for t in closed])
    
    return {
        "stats": {"total_trades": total, "win_rate_pct": round(win_rate, 2), "net_pnl": round(net_pnl, 2)},
        "all_logs": trades
    }
    
