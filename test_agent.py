import pytest
import time
from unittest.mock import MagicMock, patch
from collections import deque

from agent import PolymarketAgent

@pytest.fixture
def agent():
    with patch('agent.os.getenv', return_value='0x0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef'):
        a = PolymarketAgent(dry_run=True)
        # Mock clob client network calls
        a.client = MagicMock()
        return a

def test_calculate_sma(agent):
    product_id = "BTC-USD"
    current_time = time.time()

    # Add 10 data points spanning the last 5 minutes
    # All values = 100
    for i in range(10):
        agent.cb_history[product_id].append((current_time - (i * 30), 100.0))

    agent._calculate_sma(product_id, list(agent.cb_history[product_id]), current_time)
    assert agent.sma_5m[product_id] == 100.0

    # Add a recent high value to pull average up
    agent.cb_history[product_id].append((current_time, 200.0))
    agent._calculate_sma(product_id, list(agent.cb_history[product_id]), current_time)

    # 10 * 100 + 1 * 200 = 1200 / 11 = 109.09...
    assert round(agent.sma_5m[product_id], 2) == 109.09

def test_calculate_rsi(agent):
    product_id = "BTC-USD"
    current_time = time.time()

    # We need 15 minutes of data for 14-period RSI
    # Let's create an uptrend (steady gains)
    price = 100.0
    for i in range(15):
        # Insert a tick exactly 60 seconds apart
        ts = current_time - (15 - i) * 60
        agent.cb_history[product_id].append((ts, price))
        price += 10.0 # Gain of 10 each minute

    agent._calculate_rsi(product_id, list(agent.cb_history[product_id]), current_time, period=14)
    # Steady gains, no losses -> RSI should be 100
    assert agent.rsi_1m[product_id] == 100.0

    agent.cb_history[product_id].clear()

    # Let's create a downtrend
    price = 1000.0
    for i in range(15):
        ts = current_time - (15 - i) * 60
        agent.cb_history[product_id].append((ts, price))
        price -= 10.0 # Loss of 10 each minute

    agent._calculate_rsi(product_id, list(agent.cb_history[product_id]), current_time, period=14)
    # Steady losses, no gains -> RSI should be 0
    assert agent.rsi_1m[product_id] == 0.0

@patch('agent.logger')
def test_evaluate_market_skips_on_rsi(mock_logger, agent):
    product_id = "BTC-USD"
    market = {"up_token_id": "1", "down_token_id": "2"}

    # RSI is missing
    agent.evaluate_market(market, product_id)
    mock_logger.debug.assert_called_with("[BTC-USD] RSI not yet available")

    # RSI is overbought (> 70)
    agent.rsi_1m[product_id] = 75.0
    agent.evaluate_market(market, product_id)
    mock_logger.debug.assert_called_with("[BTC-USD] RSI=75.00 is in overbought/oversold territory. Skipping trade.")

    # RSI is oversold (< 30)
    agent.rsi_1m[product_id] = 25.0
    agent.evaluate_market(market, product_id)
    mock_logger.debug.assert_called_with("[BTC-USD] RSI=25.00 is in overbought/oversold territory. Skipping trade.")

@patch('agent.logger')
def test_evaluate_market_spread_logic(mock_logger, agent):
    product_id = "BTC-USD"
    market = {"up_token_id": "1", "down_token_id": "2"}
    agent.rsi_1m[product_id] = 50.0 # Valid RSI

    # Mock Orderbook return with wide spread (0.05)
    mock_ob = MagicMock()
    mock_ob.bids = [MagicMock(price="0.40")]
    mock_ob.asks = [MagicMock(price="0.45")]
    agent.client.get_order_book.return_value = mock_ob

    agent.evaluate_market(market, product_id)
    mock_logger.debug.assert_called_with("[BTC-USD] Spread 0.0500 is too wide. Skipping trade.")

    # Mock Orderbook return with narrow spread (0.01)
    mock_ob.bids = [MagicMock(price="0.49")]
    mock_ob.asks = [MagicMock(price="0.50")]

    # Missing cb_prices or sma_5m
    agent.evaluate_market(market, product_id)
    # Should exit silently (returning None) when data is missing, we can verify execute_trade wasn't called
    with patch.object(agent, 'execute_trade') as mock_exec:
        agent.evaluate_market(market, product_id)
        mock_exec.assert_not_called()

def test_mean_reversion_execution(agent):
    product_id = "BTC-USD"
    market = {"up_token_id": "1", "down_token_id": "2"}

    agent.rsi_1m[product_id] = 50.0

    # Narrow spread (0.01)
    mock_ob = MagicMock()
    mock_ob.bids = [MagicMock(price="0.49")]
    mock_ob.asks = [MagicMock(price="0.50")] # polymarket_up_price = 0.50
    agent.client.get_order_book.return_value = mock_ob

    # If SMA is 100, and current price is 99, it's dropped 1%
    agent.sma_5m[product_id] = 100.0
    agent.cb_prices[product_id] = 99.0

    # Deviation = -0.01
    # Fair price up = 0.5 - (-0.01 * 20) = 0.5 + 0.2 = 0.70
    # polymarket_up_price = 0.50
    # 0.70 - 0.50 = 0.20 (which is >= 0.01 threshold) -> Trade executes!

    with patch.object(agent, 'execute_trade') as mock_exec:
        agent.evaluate_market(market, product_id)
        mock_exec.assert_called_once_with(market, "UP", 0.50, 99.0, product_id)

def test_execute_trade_calculates_size(agent):
    product_id = "BTC-USD"
    market = {"up_token_id": "token_up", "down_token_id": "token_down"}

    # 1000 balance * 0.05 pct = 50 USDC
    agent.balance = 1000.0

    # Target price 0.50
    # 50 USDC / 0.50 = 100 shares

    # Should append to open_orders in dry run
    assert len(agent.open_orders) == 0
    agent.execute_trade(market, "UP", 0.50, 60000.0, product_id)

    assert len(agent.open_orders) == 1
    assert agent.open_orders[0]["token_id"] == "token_up"
    assert agent.open_orders[0]["entry_cb_price"] == 60000.0
    assert agent.open_orders[0]["mock_order"] is True
