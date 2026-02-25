from lumibot.data_sources import AlphaVantageData, DataSourceBacktesting
from lumibot.entities import Asset
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
    def __init__(
        self, datetime_start, datetime_end, config=None, api_key=None, **kwargs
    ):
        # Handle API Key / Config consistency
        if config is None:
            # Create a simple config object if api_key is passed directly
            key = api_key or kwargs.get("api_key")
            if key:
                class Config:
                    def __init__(self, k):
                        self.API_KEY = k
                config = Config(key)

        # Initialize AlphaVantageData explicitly
        AlphaVantageData.__init__(self, config=config, **kwargs)
        # Initialize DataSourceBacktesting explicitly
        DataSourceBacktesting.__init__(self, datetime_start, datetime_end, config=config, **kwargs)
        self.datetime_start = datetime_start
        self.datetime_end = datetime_end

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
        self._timestep = timestep

    def get_historical_prices(
        self, asset, length, timestep="", timeshift=None, quote=None,
        exchange=None, include_after_hours=True, **kwargs
    ):
        """
        Implementation of the abstract method required by DataSource.
        """
        # If timestep is not provided, use the class default
        if not timestep:
            timestep = self.get_timestep()
            
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
            return None

        # Parse into Bars object
        bars = self._parse_source_symbol_bars(data, asset, quote=quote, length=length)
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
                    try:
                        resp_json = json.loads(content)
                        msg = resp_json.get("Information", "") or resp_json.get("Note", "") or content[:120]
                    except Exception:
                        msg = content[:120]
                    logger.debug("AV API returned JSON (rate-limit or premium): %s", str(msg)[:80])
                    # Premium-only endpoint → signal caller to try next endpoint
                    if "premium" in msg.lower() or "premium" in content.lower():
                        return "PREMIUM"
                    # Rate-limit hit → wait and retry
                    if attempt < retries - 1:
                        logger.warning("AV rate-limit, waiting %ds before retry %d/%d", delay, attempt + 1, retries)
                        _time.sleep(delay)
                        delay *= 2
                        continue
                    return None

                df = pd.read_csv(StringIO(content))
                return df
            except Exception as e:
                logger.warning("AV fetch exception (attempt %d): %s", attempt + 1, e)
                if attempt < retries - 1:
                    _time.sleep(5)
        return None

    def _pull_source_symbol_bars(
        self, asset, timestep="minute", timeshift=None, quote=None,
        exchange=None, include_after_hours=True, **kwargs
    ):
        """
        Override _pull_source_symbol_bars to fix Alpha Vantage API calls and add caching.
        Uses TIME_SERIES_DAILY (free tier) for daily data, with automatic fallback.
        """
        symbol = asset.symbol

        # Check cache first
        cache_key = (asset, timestep)
        if cache_key in self._fixed_data_store:
            data = self._fixed_data_store[cache_key]
        else:
            data = None
            if timestep == "day":
                # Try free-tier endpoint first, then premium
                for endpoint in self._DAILY_ENDPOINTS:
                    url = (
                        f"https://www.alphavantage.co/query"
                        f"?function={endpoint}&symbol={symbol}"
                        f"&outputsize=compact&datatype=csv&apikey={self.config.API_KEY}"
                    )
                    result = self._fetch_av_csv(url, symbol)
                    if result is None:
                        continue
                    if isinstance(result, str) and result == "PREMIUM":
                        continue
                    if isinstance(result, pd.DataFrame):
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
                    f"&interval={interval}&outputsize=compact&datatype=csv&apikey={self.config.API_KEY}"
                )
                result = self._fetch_av_csv(url, symbol)
                if isinstance(result, pd.DataFrame):
                    data = result

            if data is None or (isinstance(data, pd.DataFrame) and data.empty):
                # Fallback to yfinance for daily data when AV fails (rate limit / premium)
                if timestep == "day":
                    try:
                        import yfinance as yf
                        ticker = yf.Ticker(symbol)
                        start_dt = getattr(self, "datetime_start", None) or pd.Timestamp.now() - pd.Timedelta(days=365*5)
                        end_dt = getattr(self, "datetime_end", None) or pd.Timestamp.now()
                        df = ticker.history(
                            start=start_dt,
                            end=end_dt,
                            auto_adjust=True
                        )
                        if df is not None and len(df) > 10:
                            df = df.rename(columns={
                                "Open": "open", "High": "high",
                                "Low": "low", "Close": "close", "Volume": "volume"
                            })[["open", "high", "low", "close", "volume"]]
                            if df.index.tz is None:
                                df.index = df.index.tz_localize(LUMIBOT_DEFAULT_PYTZ)
                            else:
                                df.index = df.index.tz_convert(LUMIBOT_DEFAULT_PYTZ)
                            data = df
                            self._fixed_data_store[cache_key] = data
                    except Exception as yf_err:
                        logger.debug("yfinance fallback failed: %s", yf_err)

            if data is None or (isinstance(data, pd.DataFrame) and data.empty):
                return None

            try:
                if "timestamp" in data.columns:
                    data = data.set_index("timestamp")
                elif "time" in data.columns:
                    data = data.set_index("time")

                # Convert index to datetime objects (handle both tz-naive and tz-aware)
                idx = pd.to_datetime(data.index)
                if idx.tz is None:
                    idx = idx.tz_localize(LUMIBOT_DEFAULT_PYTZ)
                else:
                    idx = idx.tz_convert(LUMIBOT_DEFAULT_PYTZ)
                data.index = idx.astype("O")
                # Sort ascending (Alpha Vantage returns descending)
                data = data.sort_index()

                self._fixed_data_store[cache_key] = data
            except Exception as e:
                logger.warning("AV _pull_source_symbol_bars exception: %s", e)
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
        """Implementation of the abstract method required by DataSource."""
        try:
            timestep = self.get_timestep()
            bars = self.get_historical_prices(
                asset, 1, timestep=timestep, quote=quote, exchange=exchange
            )
            if bars:
                return bars.get_last_price()
        except Exception as e:
            logger.debug("get_last_price error for %s: %s", asset.symbol, e)
        return None

    def get_chains(self, asset: Asset, quote: Asset = None):
        """Alpha Vantage does not support options chains"""
        return AlphaVantageData.get_chains(self, asset, quote=quote)
