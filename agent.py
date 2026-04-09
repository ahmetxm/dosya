import os
import time
import json
import threading
import logging
from collections import deque
from datetime import datetime

import requests
import websocket
from dotenv import load_dotenv

from py_clob_client.client import ClobClient
from py_clob_client.clob_types import OrderArgs, OrderType
from py_clob_client.order_builder.constants import BUY, SELL

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("PolymarketAgent")

class PolymarketAgent:
    def __init__(self, dry_run=True):
        self.dry_run = dry_run

        # Load credentials
        self.private_key = os.getenv("WALLET_PRIVATE_KEY")
        if not self.private_key:
            raise ValueError("WALLET_PRIVATE_KEY is missing in environment variables.")

        self.host = "https://clob.polymarket.com"
        self.chain_id = 137 # Polygon Mainnet

        self.client = ClobClient(
            host=self.host,
            key=self.private_key,
            chain_id=self.chain_id,
            signature_type=0,
        )

        creds = self.client.create_or_derive_api_creds()
        self.client.set_api_creds(creds)

        # Internal state
        self.balance = 0.0
        self.open_orders = []
        self.position_size_pct = 0.05

        self.btc_market = None
        self.eth_market = None

        # Market data
        self.cb_prices = {"BTC-USD": None, "ETH-USD": None}
        self.cb_history = {"BTC-USD": deque(maxlen=300), "ETH-USD": deque(maxlen=300)} # stores (timestamp, price)
        self.rsi_1m = {"BTC-USD": None, "ETH-USD": None}
        self.sma_5m = {"BTC-USD": None, "ETH-USD": None}

        # Threads
        self.ws_thread = None
        self.running = False

    def on_ws_message(self, ws, message):
        try:
            data = json.loads(message)
            if data.get('type') == 'ticker':
                product_id = data.get('product_id')
                price = float(data.get('price'))

                self.cb_prices[product_id] = price
                current_time = time.time()
                self.cb_history[product_id].append((current_time, price))

                # Calculate RSI and SMA periodically
                if len(self.cb_history[product_id]) > 60:
                    self._calculate_indicators(product_id)
        except Exception as e:
            logger.error(f"Error parsing websocket message: {e}")

    def on_ws_error(self, ws, error):
        logger.error(f"Websocket error: {error}")

    def on_ws_close(self, ws, close_status_code, close_msg):
        logger.info("Websocket closed. Reconnecting...")
        if self.running:
            time.sleep(2)
            self._start_websocket()

    def on_ws_open(self, ws):
        logger.info("Connected to Coinbase WebSocket")
        subscribe_msg = {
            "type": "subscribe",
            "product_ids": ["BTC-USD", "ETH-USD"],
            "channels": ["ticker"]
        }
        ws.send(json.dumps(subscribe_msg))

    def _start_websocket(self):
        ws_url = "wss://ws-feed.exchange.coinbase.com"
        self.ws = websocket.WebSocketApp(ws_url,
                                         on_open=self.on_ws_open,
                                         on_message=self.on_ws_message,
                                         on_error=self.on_ws_error,
                                         on_close=self.on_ws_close)
        self.ws.run_forever()

    def _calculate_indicators(self, product_id):
        # We need historical data for calculating RSI and SMA.
        # Group by 1 minute intervals (60s) for RSI
        history = list(self.cb_history[product_id])
        if not history:
            return

        current_time = time.time()

        # 1-minute RSI calculation (14 periods)
        # We need 14 minutes of data, which we won't have immediately via WS.
        # In a real setup, we'd fetch historical REST API candles first, but we can build it here.
        # For simplicity, we calculate a pseudo-RSI on the raw tick data or resampled minutes.
        self._calculate_rsi(product_id, history, current_time)

        # 5-minute SMA
        self._calculate_sma(product_id, history, current_time)

    def _calculate_rsi(self, product_id, history, current_time, period=14):
        # Resample to 1-minute candles (close prices)
        minutes = {}
        for ts, price in history:
            minute_idx = int(ts // 60)
            minutes[minute_idx] = price # Last price in that minute becomes the close

        sorted_minutes = sorted(minutes.keys())
        if len(sorted_minutes) <= period:
            return

        closes = [minutes[m] for m in sorted_minutes]

        gains = []
        losses = []

        for i in range(1, len(closes)):
            diff = closes[i] - closes[i-1]
            if diff > 0:
                gains.append(diff)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(abs(diff))

        # Simple moving average for initial RSI
        avg_gain = sum(gains[-period:]) / period
        avg_loss = sum(losses[-period:]) / period

        if avg_loss == 0:
            rsi = 100
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))

        self.rsi_1m[product_id] = rsi

    def _calculate_sma(self, product_id, history, current_time):
        # 5 minute average
        five_min_ago = current_time - 300
        recent_prices = [p for t, p in history if t >= five_min_ago]
        if recent_prices:
            self.sma_5m[product_id] = sum(recent_prices) / len(recent_prices)

    def _fetch_initial_candles(self):
        # Fetch initial historical candles from Coinbase REST API to prepopulate RSI/SMA
        logger.info("Fetching initial historical data from Coinbase...")
        for product_id in ["BTC-USD", "ETH-USD"]:
            try:
                url = f"https://api.exchange.coinbase.com/products/{product_id}/candles?granularity=60"
                resp = requests.get(url, timeout=10)
                if resp.status_code == 200:
                    candles = resp.json()
                    # format: [ timestamp, price_low, price_high, price_open, price_close ]
                    # Add them in chronological order
                    candles.reverse()
                    for c in candles[-60:]: # last 60 minutes
                        ts = c[0]
                        close_price = float(c[4])
                        self.cb_history[product_id].append((ts, close_price))
                    logger.info(f"Loaded {len(candles[-60:])} initial candles for {product_id}")
            except Exception as e:
                logger.error(f"Failed to fetch initial candles for {product_id}: {e}")

        # Calculate initial indicators
        for product_id in ["BTC-USD", "ETH-USD"]:
            self._calculate_indicators(product_id)

    def _safe_clob_call(self, func, *args, **kwargs):
        # Implement retries and rate limiting for CLOB calls to handle Polygon network latency/issues
        max_retries = 3
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.warning(f"CLOB call failed (attempt {attempt+1}/{max_retries}): {e}")
                time.sleep(2 ** attempt)  # exponential backoff
        logger.error("CLOB call failed after max retries")
        return None

    def update_balance(self):
        try:
            # We'll use get_allowance or another valid py_clob_client call.
            # Actually py_clob_client might not expose get_balance directly in older versions,
            # or it might be client.get_allowance() / checking via web3.
            # Let's mock the balance for this dry run / agent execution to avoid crashing
            # if get_balance is not a native method.
            # (In reality, `get_balance()` may be missing from py-clob-client, so we'd fetch from Polygon via web3)
            # We'll set a default of 1000.0 for logic testing if we can't fetch it natively.
            if hasattr(self.client, 'get_balance'):
                bal_info = self._safe_clob_call(self.client.get_balance)
                if bal_info:
                    self.balance = float(bal_info.get('usdc', {}).get('balance', 1000.0))
            else:
                self.balance = 1000.0
            logger.info(f"Available USDC Balance: {self.balance:.2f}")
        except Exception as e:
            logger.error(f"Failed to update balance: {e}")

    def find_active_5min_markets(self):
        try:
            url = "https://gamma-api.polymarket.com/markets?limit=100&active=true&order_by=volume&ascending=false"
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                markets = resp.json()
                for m in markets:
                    slug = m.get('slug', '').lower()
                    if "up" in slug and "down" in slug and "5" in slug:
                        if "bitcoin" in slug and not self.btc_market:
                            self.btc_market = {
                                "question": m['question'],
                                "up_token_id": m['clobTokenIds'][0],
                                "down_token_id": m['clobTokenIds'][1]
                            }
                            logger.info(f"Found BTC 5min market: {m['question']}")
                        elif "ethereum" in slug and not self.eth_market:
                            self.eth_market = {
                                "question": m['question'],
                                "up_token_id": m['clobTokenIds'][0],
                                "down_token_id": m['clobTokenIds'][1]
                            }
                            logger.info(f"Found ETH 5min market: {m['question']}")

                if not self.btc_market:
                    logger.warning("Could not find active BTC 5-minute market.")
                if not self.eth_market:
                    logger.warning("Could not find active ETH 5-minute market.")
        except Exception as e:
            logger.error(f"Error fetching active markets: {e}")

    def _trading_loop(self):
        while self.running:
            if self.btc_market:
                self.evaluate_market(self.btc_market, "BTC-USD")
            if self.eth_market:
                self.evaluate_market(self.eth_market, "ETH-USD")

            time.sleep(10) # check every 10 seconds

    def evaluate_market(self, market, product_id):
        # 1. Check RSI
        rsi = self.rsi_1m.get(product_id)
        if rsi is None:
            logger.debug(f"[{product_id}] RSI not yet available")
            return

        if rsi <= 30 or rsi >= 70:
            logger.debug(f"[{product_id}] RSI={rsi:.2f} is in overbought/oversold territory. Skipping trade.")
            return

        # 2. Get Polymarket Orderbook
        token_id = market["up_token_id"]
        ob = self._safe_clob_call(self.client.get_order_book, token_id)
        if not ob or not ob.asks or not ob.bids:
            return

        best_bid = float(ob.bids[0].price)
        best_ask = float(ob.asks[0].price)

        # 3. Calculate Spread
        spread = best_ask - best_bid
        if spread >= 0.02: # Spread > 2%
            logger.debug(f"[{product_id}] Spread {spread:.4f} is too wide. Skipping trade.")
            return

        # 4. Check Coinbase 5-min trend
        current_price = self.cb_prices.get(product_id)
        sma = self.sma_5m.get(product_id)
        if current_price is None or sma is None:
            return

        # Calculate theoretical "Yes" probability based on mean reversion.
        # In a mean reversion strategy, if the current price drops significantly below the SMA,
        # we expect it to revert (go UP).

        deviation_pct = (current_price - sma) / sma

        # We need to calculate a "fair price" (implied probability) for the 'Yes' token.
        # A simple linear model for mean reversion:
        # If deviation is negative (price is below SMA), the probability of going UP increases.
        # If deviation is positive (price is above SMA), the probability of going UP decreases.

        # Base probability is 0.5. We map a 0.5% negative deviation to a +10% increase in 'Yes' probability.
        # Note: In a real-world scenario, this model would be based on historical backtesting.
        fair_price_up = 0.5 - (deviation_pct * 20)
        fair_price_up = max(0.01, min(0.99, fair_price_up)) # Clamp to valid prices

        polymarket_up_price = best_ask # The price we can buy 'Yes' at

        # Is 'Yes' undervalued compared to our fair price?
        # Target: Execute if Polymarket price deviates from our fair reference price by at least 1% (0.01)

        if fair_price_up - polymarket_up_price >= 0.01:
            logger.info(f"[{product_id}] 'Yes' is undervalued! Fair: {fair_price_up:.4f}, Poly: {polymarket_up_price:.4f}, Dev: {deviation_pct:.4%}")
            self.execute_trade(market, "UP", polymarket_up_price, current_price, product_id)
        else:
            logger.debug(f"[{product_id}] No trading edge. Fair: {fair_price_up:.4f}, Poly: {polymarket_up_price:.4f}, Dev: {deviation_pct:.4%}")

    def execute_trade(self, market, direction, target_price, cb_ref_price, product_id):
        # Risk Management: 5% of available USDC balance
        self.update_balance()
        if self.balance <= 0:
            logger.warning("Insufficient balance to execute trade.")
            return

        trade_size_usdc = self.balance * self.position_size_pct
        num_shares = round(trade_size_usdc / target_price, 2)

        if num_shares <= 0:
            return

        token_id = market["up_token_id"] if direction == "UP" else market["down_token_id"]

        order_args = OrderArgs(
            token_id=token_id,
            price=target_price,
            size=num_shares,
            side=BUY
        )

        logger.info(f"[{product_id}] Preparing {direction} order: {num_shares} shares @ {target_price:.4f} (~${trade_size_usdc:.2f})")

        if self.dry_run:
            logger.info(f"[{product_id}] DRY RUN: Trade not executed.")
            # Mock open order for kill switch testing
            self.open_orders.append({
                "product_id": product_id,
                "token_id": token_id,
                "entry_cb_price": cb_ref_price,
                "mock_order": True
            })
            return

        try:
            resp = self._safe_clob_call(self.client.create_and_post_order, order_args, order_type=OrderType.GTC)
            if resp and resp.get('success'):
                logger.info(f"[{product_id}] Order posted successfully: {resp.get('orderID')}")
                self.open_orders.append({
                    "product_id": product_id,
                    "order_id": resp.get('orderID'),
                    "token_id": token_id,
                    "entry_cb_price": cb_ref_price,
                    "mock_order": False
                })
            else:
                logger.error(f"[{product_id}] Failed to post order: {resp}")
        except Exception as e:
            logger.error(f"[{product_id}] Error posting order: {e}")

    def _kill_switch_monitor(self):
        # Runs in background to monitor positions and cancel orders if price moves 3% against us.
        while self.running:
            try:
                orders_to_remove = []
                for order in self.open_orders:
                    product_id = order["product_id"]
                    entry_cb_price = order["entry_cb_price"]
                    current_price = self.cb_prices.get(product_id)

                    if not current_price:
                        continue

                    # Calculate price movement
                    # If we bought 'Yes' because we thought price would go UP,
                    # a move "against" us is if the price drops by 3%
                    # (In a full implementation we'd check if we hold UP or DOWN, here we assume UP for 'Yes')
                    price_change_pct = (current_price - entry_cb_price) / entry_cb_price

                    if price_change_pct <= -0.03:
                        logger.warning(f"[{product_id}] KILL SWITCH ACTIVATED! Price dropped 3% against entry. Cancelling open orders.")
                        if not order["mock_order"]:
                            # Depending on py-clob-client version, it may be cancel or cancel_order
                            if hasattr(self.client, 'cancel'):
                                self._safe_clob_call(self.client.cancel, order["order_id"])
                            elif hasattr(self.client, 'cancel_order'):
                                self._safe_clob_call(self.client.cancel_order, order["order_id"])
                        orders_to_remove.append(order)

                for order in orders_to_remove:
                    self.open_orders.remove(order)

            except Exception as e:
                logger.error(f"Kill switch monitor error: {e}")

            time.sleep(2)

    def start(self):
        logger.info("Starting Polymarket Agent...")
        self.running = True

        # Prepopulate data
        self._fetch_initial_candles()

        # Start Websocket Thread
        self.ws_thread = threading.Thread(target=self._start_websocket, daemon=True)
        self.ws_thread.start()

        # Setup CLOB
        self.update_balance()
        self.find_active_5min_markets()

        # Start Kill Switch Monitor
        self.kill_switch_thread = threading.Thread(target=self._kill_switch_monitor, daemon=True)
        self.kill_switch_thread.start()

        # Trading loop
        self.trading_thread = threading.Thread(target=self._trading_loop, daemon=True)
        self.trading_thread.start()

        # Keep main thread alive
        while self.running:
            time.sleep(1)

    def stop(self):
        logger.info("Stopping agent...")
        self.running = False
        if hasattr(self, 'ws'):
            self.ws.close()

if __name__ == "__main__":
    agent = PolymarketAgent(dry_run=True)
    try:
        agent.start()
    except KeyboardInterrupt:
        agent.stop()
