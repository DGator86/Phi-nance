from lumibot.data_sources import AlphaVantageData, DataSourceBacktesting
from lumibot.entities import Asset, Bars
from lumibot.tools.lumibot_logger import get_logger
import pandas as pd
import requests
import json
import time as _time
from io import StringIO
from lumibot.constants import LUMIBOT_DEFAULT_PYTZ

logger = get_logger(__name__)

class AlphaVantageFixedDataSource(DataSourceBacktesting, AlphaVantageData):
    """
    A fixed version of AlphaVantageBacktesting that implements the required
    abstract methods for Lumibot backtesting.
    """
    def __init__(self, datetime_start, datetime_end, config=None, api_key=None, **kwargs):
        # Handle API Key / Config consistency
        if config is None:
            # Create a simple config object if api_key is passed directly
            key = api_key or kwargs.get("api_key")
            if key:
                class Config:
                    def __init__(self, k):
                        self.API_KEY = k
                config = Config(key)

        print(f"!!!! AV FIXED INSTANTIATED !!!! config={config}")
        
        # Initialize AlphaVantageData explicitly
        AlphaVantageData.__init__(self, config=config, **kwargs)
        # Initialize DataSourceBacktesting explicitly
        DataSourceBacktesting.__init__(self, datetime_start, datetime_end, config=config, **kwargs)
        
        # Extract timestep from kwargs if present (e.g. from run_backtest)
        timestep = kwargs.get("timestep")
        self._timestep = timestep if timestep else getattr(
            self, '_timestep', 'day'
        )
        
        # Ensure self.config is set (AlphaVantageData.__init__ does this, but being safe)
        if config:
            self.config = config
        
        # Initialize an improved data store that accounts for timestep
        self._fixed_data_store = {}

    def set_timestep(self, timestep):
        print(f"!!!! AV FIXED set_timestep({timestep}) !!!!")
        self._timestep = timestep

    def get_historical_prices(self, asset, length, timestep="", timeshift=None, quote=None, exchange=None, include_after_hours=True, **kwargs):
        """
        Implementation of the abstract method required by DataSource.
        """
        # If timestep is not provided, use the class default
        if not timestep:
            timestep = self.get_timestep()
            
        print(f"!!!! AV FIXED get_historical_prices for {asset.symbol} length={length} timestep={timestep} !!!!")
        
        data = self._pull_source_symbol_bars(
            asset, 
            timestep=timestep, 
            timeshift=timeshift, 
            quote=quote, 
            exchange=exchange, 
            include_after_hours=include_after_hours,
            **kwargs
        )
        
        if data is None or data.empty:
            print(f"!!!! AV FIXED get_historical_prices NO DATA for {asset.symbol} !!!!")
            return None
            
        # Parse into Bars object
        bars = self._parse_source_symbol_bars(data, asset, quote=quote, length=length)
        print(f"!!!! AV FIXED get_historical_prices success for {asset.symbol}, bars length={len(bars)} !!!!")
        return bars

    # Daily endpoint candidates: ADJUSTED requires premium; fall back to basic daily.
    _DAILY_ENDPOINTS = [
        "TIME_SERIES_DAILY",           # free tier
        "TIME_SERIES_DAILY_ADJUSTED",  # premium fallback (in case key is premium)
    ]

    def _fetch_av_csv(self, url: str, symbol: str, retries: int = 3) -> "pd.DataFrame | None":
        """
        Fetch CSV from Alpha Vantage with retry on rate-limit (JSON) responses.
        Returns a DataFrame or None.
        """
        delay = 15  # seconds to wait on rate-limit hit
        for attempt in range(retries):
            try:
                response = requests.get(url, timeout=30)
                content = response.text
                if content.strip().startswith("{"):
                    import json
                    try:
                        msg = json.loads(content).get("Information", "") or json.loads(content).get("Note", "")
                    except Exception:
                        msg = content[:120]
                    print(f"!!!! AV FIXED API returned JSON instead of CSV: {content[:100]}... !!!!")
                    # Premium-only endpoint → signal caller to try next endpoint
                    if "premium" in msg.lower() or "premium" in content.lower():
                        return "PREMIUM"
                    # Rate-limit hit → wait and retry
                    if attempt < retries - 1:
                        print(f"!!!! AV FIXED rate-limit hit, waiting {delay}s before retry {attempt+1}/{retries} !!!!")
                        _time.sleep(delay)
                        delay *= 2
                        continue
                    return None

                df = pd.read_csv(StringIO(content))
                return df
            except Exception as e:
                print(f"!!!! AV FIXED _fetch_av_csv EXCEPTION (attempt {attempt+1}): {e} !!!!")
                if attempt < retries - 1:
                    _time.sleep(5)
        return None

    def _pull_source_symbol_bars(self, asset, timestep="minute", timeshift=None, quote=None, exchange=None, include_after_hours=True, **kwargs):
        """
        Override _pull_source_symbol_bars to fix Alpha Vantage API calls and add caching.
        Uses TIME_SERIES_DAILY (free tier) for daily data, with automatic fallback.
        """
        symbol = asset.symbol
        print(f"!!!! AV FIXED _pull_source_symbol_bars for {symbol} timestep={timestep} !!!!")

        # Check cache first
        cache_key = (asset, timestep)
        if cache_key in self._fixed_data_store:
            print(f"!!!! AV FIXED cache hit for {symbol} {timestep} !!!!")
            data = self._fixed_data_store[cache_key]
        else:
            data = None
            if timestep == "day":
                # Try free-tier endpoint first, then premium
                for endpoint in self._DAILY_ENDPOINTS:
                    url = (
                        f"https://www.alphavantage.co/query"
                        f"?function={endpoint}&symbol={symbol}"
                        f"&outputsize=full&datatype=csv&apikey={self.config.API_KEY}"
                    )
                    print(f"!!!! AV FIXED API URL: {url.replace(self.config.API_KEY, 'SECRET')} !!!!")
                    result = self._fetch_av_csv(url, symbol)
                    if result is None:
                        continue
                    if result == "PREMIUM":
                        print(f"!!!! AV FIXED {endpoint} requires premium, trying next !!!!")
                        continue
                    data = result
                    break
            else:
                interval_map = {
                    "minute": "1min",
                    "5min": "5min",
                    "15min": "15min",
                    "30min": "30min",
                    "hour": "60min",
                }
                interval = interval_map.get(timestep, "1min")
                url = (
                    f"https://www.alphavantage.co/query"
                    f"?function=TIME_SERIES_INTRADAY&symbol={symbol}"
                    f"&interval={interval}&outputsize=full&datatype=csv&apikey={self.config.API_KEY}"
                )
                print(f"!!!! AV FIXED API URL: {url.replace(self.config.API_KEY, 'SECRET')} !!!!")
                result = self._fetch_av_csv(url, symbol)
                if result is not None and result != "PREMIUM":
                    data = result

            if data is None or (isinstance(data, pd.DataFrame) and data.empty):
                print(f"!!!! AV FIXED get_historical_prices NO DATA for {symbol} !!!!")
                return None

            try:
                if "timestamp" in data.columns:
                    data = data.set_index("timestamp")
                elif "time" in data.columns:
                    data = data.set_index("time")

                # Convert index to datetime objects
                data.index = pd.to_datetime(data.index).tz_localize(tz=LUMIBOT_DEFAULT_PYTZ).astype("O")
                # Sort ascending (Alpha Vantage returns descending)
                data = data.sort_index()

                self._fixed_data_store[cache_key] = data
                print(f"!!!! AV FIXED cache populated for {symbol} {timestep}, rows={len(data)} !!!!")
            except Exception as e:
                print(f"!!!! AV FIXED _pull_source_symbol_bars EXCEPTION: {e} !!!!")
                return None

        # Filter by current simulation time (self._datetime from DataSourceBacktesting)
        if hasattr(self, "_datetime") and self._datetime is not None:
            data = data[data.index <= self._datetime]

        # Get length if provided
        length = kwargs.get("length")
        if length:
            data = data.tail(length)

        return data

    def get_last_price(self, asset, quote=None, exchange=None):
        """
        Implementation of the abstract method required by DataSource.
        """
        print(f"!!!! AV FIXED get_last_price for {asset.symbol} !!!!")
        try:
            # Try to use the same timestep as the backtest
            timestep = self.get_timestep()
            print(f"!!!! AV FIXED get_last_price using timestep={timestep} !!!!")
            
            # Explicitly pass timestep to get_historical_prices
            bars = self.get_historical_prices(asset, 1, timestep=timestep, quote=quote, exchange=exchange)
            if bars:
                last_price = bars.get_last_price()
                print(f"!!!! AV FIXED get_last_price for {asset.symbol} = {last_price} !!!!")
                return last_price
        except Exception as e:
            print(f"!!!! AV FIXED get_last_price ERROR: {e} !!!!")
            import traceback
            traceback.print_exc()
        return None

    def get_chains(self, asset: Asset, quote: Asset = None):
        """Alpha Vantage does not support options chains"""
        return AlphaVantageData.get_chains(self, asset, quote=quote)
