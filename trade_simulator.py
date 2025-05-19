import json
import time
import threading
import logging
import numpy as np
import pandas as pd
import websocket
import tkinter as tk
from tkinter import ttk
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.stats import linregress
from sklearn.linear_model import LinearRegression, LogisticRegression

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('TradeSimulator')

class OrderBook:
    """
    Class to maintain a local copy of the L2 order book
    """
    def __init__(self):
        self.asks = []  # List of [price, quantity] for asks
        self.bids = []  # List of [price, quantity] for bids
        self.timestamp = None
        self.exchange = None
        self.symbol = None
        self.lock = threading.Lock()
        self.historical_data = []  # Store historical snapshots for regression models
        self.max_history = 100  # Maximum number of historical snapshots to keep

    def update(self, data):
        """Update the order book with new data"""
        with self.lock:
            self.timestamp = data.get("timestamp")
            self.exchange = data.get("exchange")
            self.symbol = data.get("symbol")
            self.asks = [[float(price), float(qty)] for price, qty in data.get("asks", [])]
            self.bids = [[float(price), float(qty)] for price, qty in data.get("bids", [])]
            
            # Save a snapshot for historical analysis
            if len(self.historical_data) >= self.max_history:
                self.historical_data.pop(0)  # Remove oldest entry
            
            # Create a snapshot with midpoint price and spreads
            midpoint = self.get_midpoint_price()
            spread = self.get_spread()
            depth = self.get_market_depth(100)  # $100 depth
            
            self.historical_data.append({
                'timestamp': self.timestamp,
                'midpoint': midpoint,
                'spread': spread,
                'depth': depth,
                'best_ask': self.asks[0][0] if self.asks else None,
                'best_bid': self.bids[0][0] if self.bids else None
            })

    def get_midpoint_price(self):
        """Calculate the midpoint price"""
        if not self.asks or not self.bids:
            return None
        best_ask = self.asks[0][0]
        best_bid = self.bids[0][0]
        return (best_ask + best_bid) / 2

    def get_spread(self):
        """Calculate the bid-ask spread"""
        if not self.asks or not self.bids:
            return None
        best_ask = self.asks[0][0]
        best_bid = self.bids[0][0]
        return best_ask - best_bid

    def get_market_depth(self, usd_depth):
        """Calculate the market depth for a given USD amount"""
        if not self.asks or not self.bids:
            return (0, 0)
        
        # Calculate ask depth
        ask_depth = 0
        ask_value = 0
        for price, qty in self.asks:
            value = price * qty
            if ask_value + value > usd_depth:
                # Calculate partial quantity needed
                remaining = usd_depth - ask_value
                ask_depth += remaining / price
                break
            ask_depth += qty
            ask_value += value
            if ask_value >= usd_depth:
                break
        
        # Calculate bid depth
        bid_depth = 0
        bid_value = 0
        for price, qty in self.bids:
            value = price * qty
            if bid_value + value > usd_depth:
                # Calculate partial quantity needed
                remaining = usd_depth - bid_value
                bid_depth += remaining / price
                break
            bid_depth += qty
            bid_value += value
            if bid_value >= usd_depth:
                break
                
        return (ask_depth, bid_depth)

    def get_impact_price(self, side, quantity):
        """
        Calculate the average execution price after market impact
        side: 'buy' or 'sell'
        quantity: amount to execute
        """
        if not self.asks or not self.bids:
            return None
            
        if side.lower() == 'buy':
            levels = self.asks
        else:
            levels = self.bids
            
        executed_qty = 0
        total_cost = 0
        
        for price, qty in levels:
            if executed_qty + qty >= quantity:
                # Partial execution at this level
                remaining = quantity - executed_qty
                total_cost += price * remaining
                executed_qty = quantity  # We've filled the order
                break
            else:
                # Full execution at this level
                executed_qty += qty
                total_cost += price * qty
                
        if executed_qty < quantity:
            logger.warning(f"Not enough liquidity to execute {quantity} units")
            return None
            
        return total_cost / quantity

class AlmgrenChrissModel:
    """
    Implementation of Almgren-Chriss market impact model
    """
    def __init__(self):
        # Default parameters based on literature
        self.sigma = 0.3  # Daily volatility
        self.eta = 2.5e-6  # Temporary impact factor
        self.gamma = 2.5e-6  # Permanent impact factor
        
    def set_volatility(self, sigma):
        """Set the volatility parameter"""
        self.sigma = sigma
        
    def calculate_market_impact(self, quantity, current_price, market_cap=None, daily_volume=None, time_horizon=1):
        """
        Calculate market impact using Almgren-Chriss model
        
        Parameters:
        - quantity: size of the order
        - current_price: current asset price
        - market_cap: market capitalization (optional)
        - daily_volume: average daily volume (optional)
        - time_horizon: execution time in days (default: 1)
        
        Returns:
        - tuple of (temporary_impact, permanent_impact, total_impact)
        """
        order_value = quantity * current_price
        
        # Calculate temporary impact (immediate price move)
        # η * σ * |x| / √T
        temp_impact = self.eta * self.sigma * order_value / np.sqrt(time_horizon)
        
        # Calculate permanent impact (price change after execution)
        # γ * |x|
        perm_impact = self.gamma * order_value
        
        total_impact = temp_impact + perm_impact
        
        # Return components and total
        return (temp_impact, perm_impact, total_impact)
        
    def optimal_execution_trajectory(self, quantity, time_steps):
        """
        Calculate optimal execution trajectory
        """
        # Almgren-Chriss suggests a trajectory that minimizes the tradeoff
        # between market impact and price risk
        
        # For risk-neutral trader, the optimal trajectory is linear
        # For risk-averse trader, it's front-loaded
        
        # Simple implementation: linearly decreasing trajectory
        remaining = np.linspace(quantity, 0, time_steps)
        trades = np.diff(np.append(remaining, 0))
        
        return np.abs(trades)

class RegressionModels:
    """
    Class implementing regression models for slippage and market impact estimation
    """
    def __init__(self):
        self.slippage_model = LinearRegression()
        self.maker_taker_model = LogisticRegression()
        self.is_fitted_slippage = False
        self.is_fitted_maker_taker = False
        
    def prepare_features(self, orderbook_history, quantity):
        """Prepare features for regression models"""
        if not orderbook_history or len(orderbook_history) < 10:
            return None
            
        # Extract features from orderbook history
        features = []
        for snapshot in orderbook_history:
            if all(key in snapshot for key in ['midpoint', 'spread', 'depth']):
                # Create feature vector
                feature = [
                    snapshot['spread'],  # Bid-ask spread
                    snapshot['depth'][0],  # Ask depth
                    snapshot['depth'][1],  # Bid depth
                    quantity,  # Order quantity
                ]
                features.append(feature)
                
        return np.array(features) if features else None
        
    def fit_slippage_model(self, orderbook_history, quantity, observed_slippage):
        """
        Fit linear regression model for slippage prediction
        In a real scenario, observed_slippage would come from historical executions
        For this simulation, we'll generate synthetic data
        """
        X = self.prepare_features(orderbook_history, quantity)
        if X is None:
            return False
            
        # For simulation, we'll create synthetic slippage data
        # In practice, this would be historical execution data
        y = np.random.normal(0.001 * quantity, 0.0005 * quantity, size=X.shape[0])
        
        try:
            self.slippage_model.fit(X, y)
            self.is_fitted_slippage = True
            return True
        except Exception as e:
            logger.error(f"Error fitting slippage model: {e}")
            return False
    
    def predict_slippage(self, orderbook_history, quantity):
        """Predict slippage for a given order quantity"""
        if not self.is_fitted_slippage:
            # If model isn't fitted, use a simple estimate
            # Average spread * quantity / 2 as a baseline
            if orderbook_history and len(orderbook_history) > 0:
                avg_spread = np.mean([snapshot.get('spread', 0) for snapshot in orderbook_history if 'spread' in snapshot])
                return (avg_spread * quantity / 2)
            return 0.001 * quantity  # Default 0.1% slippage
            
        X = self.prepare_features(orderbook_history, quantity)
        if X is None or X.shape[0] == 0:
            return 0.001 * quantity
            
        # Use the last set of features for prediction
        slippage = self.slippage_model.predict([X[-1]])
        return max(0, slippage[0])  # Ensure non-negative slippage
        
    def fit_maker_taker_model(self, orderbook_history, quantities, is_maker):
        """
        Fit logistic regression to predict maker/taker proportion
        For simulation, we'll use synthetic data
        """
        X = self.prepare_features(orderbook_history, np.mean(quantities))
        if X is None:
            return False
            
        # Create synthetic maker/taker data
        # In practice, this would come from historical executions
        y = np.random.binomial(1, 0.3, size=X.shape[0])  # 30% chance of being a maker
        
        try:
            self.maker_taker_model.fit(X, y)
            self.is_fitted_maker_taker = True
            return True
        except Exception as e:
            logger.error(f"Error fitting maker/taker model: {e}")
            return False
            
    def predict_maker_taker(self, orderbook_history, quantity):
        """Predict maker/taker proportion for a given order quantity"""
        if not self.is_fitted_maker_taker:
            # If model isn't fitted, assume 20% maker / 80% taker for market orders
            return 0.2
            
        X = self.prepare_features(orderbook_history, quantity)
        if X is None or X.shape[0] == 0:
            return 0.2
            
        # Use the last set of features for prediction
        prob_maker = self.maker_taker_model.predict_proba([X[-1]])
        return prob_maker[0][1]  # Probability of being a maker

class FeeModel:
    """
    Fee model based on exchange fee tiers
    """
    def __init__(self):
        # OKX fee tiers (example)
        # In a real implementation, these would be retrieved from exchange docs or API
        self.fee_tiers = {
            "OKX": {
                "VIP0": {"maker": 0.0010, "taker": 0.0015},
                "VIP1": {"maker": 0.0008, "taker": 0.0010},
                "VIP2": {"maker": 0.0006, "taker": 0.0008},
                "VIP3": {"maker": 0.0004, "taker": 0.0006},
                "VIP4": {"maker": 0.0002, "taker": 0.0004},
                "VIP5": {"maker": 0.0000, "taker": 0.0002}
            }
        }
        
    def get_fees(self, exchange, tier, order_value, maker_proportion):
        """
        Calculate fees based on exchange, tier, and maker/taker proportion
        
        Parameters:
        - exchange: Exchange name
        - tier: Fee tier (e.g., "VIP0")
        - order_value: Total order value in quote currency
        - maker_proportion: Proportion executed as maker (0.0 to 1.0)
        
        Returns:
        - Total fee amount
        """
        if exchange not in self.fee_tiers or tier not in self.fee_tiers[exchange]:
            # Default to highest fee tier if not found
            logger.warning(f"Fee tier {tier} not found for {exchange}, using default")
            tier = list(self.fee_tiers[exchange].keys())[0]
            
        maker_fee = self.fee_tiers[exchange][tier]["maker"]
        taker_fee = self.fee_tiers[exchange][tier]["taker"]
        
        # Calculate weighted average fee
        total_fee = (maker_proportion * maker_fee + (1 - maker_proportion) * taker_fee) * order_value
        
        return total_fee
        
    def get_available_tiers(self, exchange):
        """Get available fee tiers for an exchange"""
        if exchange in self.fee_tiers:
            return list(self.fee_tiers[exchange].keys())
        return []

class TradeSimulator:
    """
    Main trade simulator class that integrates all components
    """
    def __init__(self):
        self.orderbook = OrderBook()
        self.impact_model = AlmgrenChrissModel()
        self.regression_models = RegressionModels()
        self.fee_model = FeeModel()
        self.ws = None
        self.last_tick_time = None
        self.processing_times = []  # Track processing time per tick
        self.is_connected = False
        self.lock = threading.Lock()
        
    def connect_websocket(self, url):
        """Connect to WebSocket endpoint"""
        logger.info(f"Connecting to WebSocket: {url}")
        
        def on_message(ws, message):
            """Handle incoming WebSocket messages"""
            start_time = time.time()
            
            try:
                data = json.loads(message)
                self.orderbook.update(data)
                
                # Measure processing time
                processing_time = time.time() - start_time
                with self.lock:
                    self.processing_times.append(processing_time)
                    if len(self.processing_times) > 100:
                        self.processing_times.pop(0)
                
                self.last_tick_time = time.time()
                
            except Exception as e:
                logger.error(f"Error processing message: {e}")
                
        def on_error(ws, error):
            logger.error(f"WebSocket error: {error}")
            self.is_connected = False
            
        def on_close(ws, close_status_code, close_msg):
            logger.info(f"WebSocket closed: {close_status_code} - {close_msg}")
            self.is_connected = False
            
        def on_open(ws):
            logger.info("WebSocket connection established")
            self.is_connected = True
            
        # Create WebSocket connection
        self.ws = websocket.WebSocketApp(
            url,
            on_open=on_open,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close
        )
        
        # Start WebSocket in a separate thread
        wst = threading.Thread(target=self.ws.run_forever)
        wst.daemon = True
        wst.start()
        
    def disconnect_websocket(self):
        """Disconnect from WebSocket"""
        if self.ws:
            self.ws.close()
            self.is_connected = False
            
    def calculate_transaction_costs(self, exchange, asset, order_type, quantity, volatility, fee_tier):
        """
        Calculate all transaction costs based on current orderbook and parameters
        
        Parameters:
        - exchange: Exchange name
        - asset: Trading pair
        - order_type: Order type (market, limit)
        - quantity: Order quantity in USD
        - volatility: Asset volatility
        - fee_tier: Fee tier on the exchange
        
        Returns:
        - Dictionary with all cost components
        """
        with self.orderbook.lock:
            # Check if we have valid orderbook data
            if not self.orderbook.asks or not self.orderbook.bids:
                return {
                    "slippage": 0,
                    "fees": 0,
                    "market_impact": 0,
                    "net_cost": 0,
                    "maker_taker": 0,
                    "latency": 0
                }
            
            # Get current price
            current_price = self.orderbook.get_midpoint_price()
            if current_price is None:
                return {
                    "slippage": 0,
                    "fees": 0,
                    "market_impact": 0,
                    "net_cost": 0,
                    "maker_taker": 0,
                    "latency": 0
                }
                
            # Set volatility in the model
            self.impact_model.set_volatility(volatility)
            
            # Convert USD quantity to asset quantity
            asset_quantity = quantity / current_price
            
            # Calculate expected slippage
            slippage = self.regression_models.predict_slippage(
                self.orderbook.historical_data, 
                asset_quantity
            )
            
            # Predict maker/taker proportion
            maker_proportion = self.regression_models.predict_maker_taker(
                self.orderbook.historical_data,
                asset_quantity
            )
            
            # Calculate fees
            fees = self.fee_model.get_fees(
                exchange,
                fee_tier,
                quantity,  # Order value in USD
                maker_proportion
            )
            
            # Calculate market impact
            _, _, market_impact = self.impact_model.calculate_market_impact(
                asset_quantity,
                current_price
            )
            
            # Calculate net cost
            net_cost = slippage + fees + market_impact
            
            # Calculate average processing latency
            avg_latency = np.mean(self.processing_times) if self.processing_times else 0
            
            return {
                "slippage": slippage,
                "fees": fees,
                "market_impact": market_impact,
                "net_cost": net_cost,
                "maker_taker": maker_proportion,
                "latency": avg_latency * 1000  # Convert to milliseconds
            }

class TradeSimulatorUI:
    """
    User interface for the trade simulator
    """
    def __init__(self, root):
        self.root = root
        self.root.title("GoQuant Trade Simulator")
        self.root.geometry("1200x800")
        
        # Create simulator instance
        self.simulator = TradeSimulator()
        
        # Create main frames
        self.create_layout()
        
        # Initialize input fields with default values
        self.initialize_inputs()
        
        # Initialize charts
        self.initialize_charts()
        
        # Start periodic updates
        self.start_updates()
        
    def create_layout(self):
        """Create the main UI layout"""
        # Create main frames
        self.left_frame = ttk.Frame(self.root, padding=10)
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.right_frame = ttk.Frame(self.root, padding=10)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Left frame - Input parameters
        ttk.Label(self.left_frame, text="Input Parameters", font=('Arial', 16, 'bold')).pack(anchor=tk.W, pady=(0, 10))
        
        # Create input fields
        self.create_input_fields()
        
        # Connection status indicator
        self.status_frame = ttk.Frame(self.left_frame, padding=5)
        self.status_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(self.status_frame, text="Connection Status:").pack(side=tk.LEFT)
        self.status_label = ttk.Label(self.status_frame, text="Disconnected", foreground="red")
        self.status_label.pack(side=tk.LEFT, padx=5)
        
        # Connect button
        self.connect_button = ttk.Button(self.left_frame, text="Connect", command=self.toggle_connection)
        self.connect_button.pack(anchor=tk.W, pady=5)
        
        # Calculate button
        self.calculate_button = ttk.Button(self.left_frame, text="Calculate Costs", command=self.calculate_costs)
        self.calculate_button.pack(anchor=tk.W, pady=5)
        
        # Right frame - Output parameters
        ttk.Label(self.right_frame, text="Output Parameters", font=('Arial', 16, 'bold')).pack(anchor=tk.W, pady=(0, 10))
        
        # Create output fields
        self.create_output_fields()
        
        # Create charts area
        self.charts_frame = ttk.Frame(self.right_frame, padding=5)
        self.charts_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
    def create_input_fields(self):
        """Create input parameter fields"""
        # Exchange
        exchange_frame = ttk.Frame(self.left_frame, padding=5)
        exchange_frame.pack(fill=tk.X, pady=2)
        
        ttk.Label(exchange_frame, text="Exchange:").pack(side=tk.LEFT)
        self.exchange_var = tk.StringVar()
        exchange_cb = ttk.Combobox(exchange_frame, textvariable=self.exchange_var, state="readonly")
        exchange_cb['values'] = ('OKX',)
        exchange_cb.pack(side=tk.RIGHT, padx=5, fill=tk.X, expand=True)
        
        # Asset
        asset_frame = ttk.Frame(self.left_frame, padding=5)
        asset_frame.pack(fill=tk.X, pady=2)
        
        ttk.Label(asset_frame, text="Asset:").pack(side=tk.LEFT)
        self.asset_var = tk.StringVar()
        asset_cb = ttk.Combobox(asset_frame, textvariable=self.asset_var, state="readonly")
        asset_cb['values'] = ('BTC-USDT', 'ETH-USDT', 'SOL-USDT')
        asset_cb.pack(side=tk.RIGHT, padx=5, fill=tk.X, expand=True)
        
        # Order Type
        order_frame = ttk.Frame(self.left_frame, padding=5)
        order_frame.pack(fill=tk.X, pady=2)
        
        ttk.Label(order_frame, text="Order Type:").pack(side=tk.LEFT)
        self.order_type_var = tk.StringVar()
        order_cb = ttk.Combobox(order_frame, textvariable=self.order_type_var, state="readonly")
        order_cb['values'] = ('Market',)
        order_cb.pack(side=tk.RIGHT, padx=5, fill=tk.X, expand=True)
        
        # Quantity
        qty_frame = ttk.Frame(self.left_frame, padding=5)
        qty_frame.pack(fill=tk.X, pady=2)
        
        ttk.Label(qty_frame, text="Quantity (USD):").pack(side=tk.LEFT)
        self.quantity_var = tk.StringVar()
        ttk.Entry(qty_frame, textvariable=self.quantity_var).pack(side=tk.RIGHT, padx=5, fill=tk.X, expand=True)
        
        # Volatility
        vol_frame = ttk.Frame(self.left_frame, padding=5)
        vol_frame.pack(fill=tk.X, pady=2)
        
        ttk.Label(vol_frame, text="Volatility:").pack(side=tk.LEFT)
        self.volatility_var = tk.StringVar()
        ttk.Entry(vol_frame, textvariable=self.volatility_var).pack(side=tk.RIGHT, padx=5, fill=tk.X, expand=True)
        
        # Fee Tier
        fee_frame = ttk.Frame(self.left_frame, padding=5)
        fee_frame.pack(fill=tk.X, pady=2)
        
        ttk.Label(fee_frame, text="Fee Tier:").pack(side=tk.LEFT)
        self.fee_tier_var = tk.StringVar()
        fee_cb = ttk.Combobox(fee_frame, textvariable=self.fee_tier_var, state="readonly")
        fee_cb['values'] = ('VIP0', 'VIP1', 'VIP2', 'VIP3', 'VIP4', 'VIP5')
        fee_cb.pack(side=tk.RIGHT, padx=5, fill=tk.X, expand=True)
        
    def create_output_fields(self):
        """Create output parameter fields"""
        # Create a frame for output fields
        output_container = ttk.Frame(self.right_frame, padding=5)
        output_container.pack(fill=tk.X, pady=5)
        
        # Expected Slippage
        slip_frame = ttk.Frame(output_container, padding=(0, 5))
        slip_frame.pack(fill=tk.X)
        
        ttk.Label(slip_frame, text="Expected Slippage (USD):").pack(side=tk.LEFT)
        self.slippage_var = tk.StringVar(value="0.00")
        ttk.Label(slip_frame, textvariable=self.slippage_var, font=('Arial', 10, 'bold')).pack(side=tk.RIGHT)
        
        # Expected Fees
        fees_frame = ttk.Frame(output_container, padding=(0, 5))
        fees_frame.pack(fill=tk.X)
        
        ttk.Label(fees_frame, text="Expected Fees (USD):").pack(side=tk.LEFT)
        self.fees_var = tk.StringVar(value="0.00")
        ttk.Label(fees_frame, textvariable=self.fees_var, font=('Arial', 10, 'bold')).pack(side=tk.RIGHT)
        
        # Market Impact
        impact_frame = ttk.Frame(output_container, padding=(0, 5))
        impact_frame.pack(fill=tk.X)
        
        ttk.Label(impact_frame, text="Market Impact (USD):").pack(side=tk.LEFT)
        self.impact_var = tk.StringVar(value="0.00")
        ttk.Label(impact_frame, textvariable=self.impact_var, font=('Arial', 10, 'bold')).pack(side=tk.RIGHT)
        
        # Net Cost
        cost_frame = ttk.Frame(output_container, padding=(0, 5))
        cost_frame.pack(fill=tk.X)
        
        ttk.Label(cost_frame, text="Net Cost (USD):").pack(side=tk.LEFT)
        self.net_cost_var = tk.StringVar(value="0.00")
        ttk.Label(cost_frame, textvariable=self.net_cost_var, font=('Arial', 10, 'bold')).pack(side=tk.RIGHT)
        
        # Maker/Taker proportion
        mt_frame = ttk.Frame(output_container, padding=(0, 5))
        mt_frame.pack(fill=tk.X)
        
        ttk.Label(mt_frame, text="Maker/Taker Proportion:").pack(side=tk.LEFT)
        self.maker_taker_var = tk.StringVar(value="0.00")
        ttk.Label(mt_frame, textvariable=self.maker_taker_var, font=('Arial', 10, 'bold')).pack(side=tk.RIGHT)
        
        # Internal Latency
        lat_frame = ttk.Frame(output_container, padding=(0, 5))
        lat_frame.pack(fill=tk.X)
        
        ttk.Label(lat_frame, text="Internal Latency (ms):").pack(side=tk.LEFT)
        self.latency_var = tk.StringVar(value="0.00")
        ttk.Label(lat_frame, textvariable=self.latency_var, font=('Arial', 10, 'bold')).pack(side=tk.RIGHT)
        
    def initialize_inputs(self):
        """Initialize input fields with default values"""
        self.exchange_var.set('OKX')
        self.asset_var.set('BTC-USDT')
        self.order_type_var.set('Market')
        self.quantity_var.set('100')
        self.volatility_var.set('0.03')  # 3% daily volatility
        self.fee_tier_var.set('VIP0')
        
    def initialize_charts(self):
        """Initialize charts for visualization"""
        # Create matplotlib figure for charts
        self.fig = plt.Figure(figsize=(6, 8), dpi=100)
        
        # Orderbook depth chart
        self.depth_ax = self.fig.add_subplot(3, 1, 1)
        self.depth_ax.set_title('Order Book Depth')
        self.depth_ax.set_xlabel('Price')
        self.depth_ax.set_ylabel('Quantity')
        
        # Market impact chart
        self.impact_ax = self.fig.add_subplot(3, 1, 2)
        self.impact_ax.set_title('Market Impact vs Order Size')
        self.impact_ax.set_xlabel('Order Size (USD)')
        self.impact_ax.set_ylabel('Impact (USD)')
        
        # Latency chart
        self.latency_ax = self.fig.add_subplot(3, 1, 3)
        self.latency_ax.set_title('Processing Latency')
        self.latency_ax.set_xlabel('Tick')
        self.latency_ax.set_ylabel('Latency (ms)')
        
        # Add figure to tkinter window
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.charts_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def toggle_connection(self):
        """Connect or disconnect from WebSocket"""
        if not self.simulator.is_connected:
            # Connect to WebSocket
            url = f"wss://ws.gomarket-cpp.goquant.io/ws/l2-orderbook/okx/{self.asset_var.get()}"
            self.simulator.connect_websocket(url)
            self.connect_button.configure(text="Disconnect")
            self.status_label.configure(text="Connecting...", foreground="orange")
        else:
            # Disconnect WebSocket
            self.simulator.disconnect_websocket()
            self.connect_button.configure(text="Connect")
            self.status_label.configure(text="Disconnected", foreground="red")
            
    def update_status(self):
        """Update connection status indicator"""
        if self.simulator.is_connected:
            self.status_label.configure(text="Connected", foreground="green")
        else:
            self.status_label.configure(text="Disconnected", foreground="red")
            
    def calculate_costs(self):
        """Calculate transaction costs and update output fields"""
        try:
            # Parse input values
            exchange = self.exchange_var.get()
            asset = self.asset_var.get()
            order_type = self.order_type_var.get().lower()
            quantity = float(self.quantity_var.get())
            volatility = float(self.volatility_var.get())
            fee_tier = self.fee_tier_var.get()
            
            # Calculate costs
            costs = self.simulator.calculate_transaction_costs(
                exchange, asset, order_type, quantity, volatility, fee_tier
            )
            
            # Update output fields
            self.slippage_var.set(f"{costs['slippage']:.4f}")
            self.fees_var.set(f"{costs['fees']:.4f}")
            self.impact_var.set(f"{costs['market_impact']:.4f}")
            self.net_cost_var.set(f"{costs['net_cost']:.4f}")
            self.maker_taker_var.set(f"{costs['maker_taker']:.2f}")
            self.latency_var.set(f"{costs['latency']:.2f}")
            
            # Update charts
            self.update_charts()
            
        except Exception as e:
            logger.error(f"Error calculating costs: {e}")
            # Show error message
            tk.messagebox.showerror("Error", f"Error calculating costs: {e}")
            
    def update_charts(self):
        """Update charts with current data"""
        try:
            # Clear previous plots
            self.depth_ax.clear()
            self.impact_ax.clear()
            self.latency_ax.clear()
            
            with self.simulator.orderbook.lock:
                # Plot order book depth
                if self.simulator.orderbook.asks and self.simulator.orderbook.bids:
                    # Get asks (sell orders)
                    ask_prices = [price for price, _ in self.simulator.orderbook.asks[:10]]
                    ask_quantities = [qty for _, qty in self.simulator.orderbook.asks[:10]]
                    
                    # Get bids (buy orders)
                    bid_prices = [price for price, _ in self.simulator.orderbook.bids[:10]]
                    bid_quantities = [qty for _, qty in self.simulator.orderbook.bids[:10]]
                    
                    # Plot
                    self.depth_ax.bar(ask_prices, ask_quantities, color='red', alpha=0.6, label='Asks')
                    self.depth_ax.bar(bid_prices, bid_quantities, color='green', alpha=0.6, label='Bids')
                    self.depth_ax.legend()
                    self.depth_ax.set_title('Order Book Depth')
                
                # Plot market impact for different order sizes
                try:
                    order_sizes = np.linspace(50, 500, 10)  # 50 to 500 USD
                    impacts = []
                    
                    current_price = self.simulator.orderbook.get_midpoint_price()
                    if current_price:
                        volatility = float(self.volatility_var.get())
                        for size in order_sizes:
                            # Convert USD to quantity
                            qty = size / current_price
                            
                            # Calculate market impact
                            _, _, impact = self.simulator.impact_model.calculate_market_impact(
                                qty, current_price, volatility=volatility
                            )
                            impacts.append(impact)
                            
                        self.impact_ax.plot(order_sizes, impacts, 'b-', marker='o')
                        self.impact_ax.set_title('Market Impact vs Order Size')
                except Exception as e:
                    logger.error(f"Error plotting market impact: {e}")
                
                # Plot latency data
                if self.simulator.processing_times:
                    latencies = np.array(self.simulator.processing_times) * 1000  # Convert to ms
                    self.latency_ax.plot(range(len(latencies)), latencies, 'g-', marker='.')
                    self.latency_ax.set_title(f'Processing Latency (Avg: {np.mean(latencies):.2f} ms)')
            
            # Redraw canvas
            self.fig.tight_layout()
            self.canvas.draw()
            
        except Exception as e:
            logger.error(f"Error updating charts: {e}")
            
    def start_updates(self):
        """Start periodic UI updates"""
        def update_loop():
            self.update_status()
            
            # Update charts periodically
            if self.simulator.is_connected:
                # Fit regression models if we have enough data
                if (len(self.simulator.orderbook.historical_data) > 20 and 
                    not self.simulator.regression_models.is_fitted_slippage):
                    # Try to fit models with synthetic data
                    self.simulator.regression_models.fit_slippage_model(
                        self.simulator.orderbook.historical_data, 
                        float(self.quantity_var.get()),
                        None  # No observed slippage for simulation
                    )
                    
                    self.simulator.regression_models.fit_maker_taker_model(
                        self.simulator.orderbook.historical_data,
                        [float(self.quantity_var.get())],
                        None  # No observed maker/taker data for simulation
                    )
                
                # Update output values
                self.calculate_costs()
            
            # Schedule next update
            self.root.after(1000, update_loop)
            
        # Start the update loop
        self.root.after(1000, update_loop)
        
def main():
    """Main entry point for the application"""
    root = tk.Tk()
    app = TradeSimulatorUI(root)
    root.mainloop()
    
if __name__ == "__main__":
    # Enable websocket trace for debugging
    websocket.enableTrace(True)
    main()