import os
import math
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field, field_validator, model_validator
import yfinance as yf
import pandas as pd
import numpy as np


# =========================================================
# FASTAPI APP
# =========================================================

app = FastAPI(
    title="QuantX Financial Intelligence API",
    description="Multi-Asset Quantitative Analysis and Backtesting Platform with Robust Validation",
    version="1.1"
)


# =========================================================
# CORS
# =========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# CONSTANTS & ALLOWED VALUES
# =========================================================

QUANTX_API_KEY = os.getenv("QUANTX_API_KEY", "rc_43a33fc68f4641e598173a55da96b75af2d580ba0ad3b47a20b48c00b0582503")

ASSETS = {
    "gold": "GC=F",
    "bitcoin": "BTC-USD",
    "nvidia": "NVDA"
}

ALLOWED_ASSETS = set(ASSETS.keys())
ALLOWED_STRATEGIES = {"sma", "ema", "momentum", "mean_reversion"}
ALLOWED_PERIODS = {"1mo", "3mo", "6mo", "1y", "2y", "3y", "5y", "10y", "ytd", "max"}


# =========================================================
# API KEY AUTHENTICATION HELPER
# =========================================================

def check_api_key(request: Request) -> bool:
    """Verifies that if an API key is provided, it matches the configured QuantX API key."""
    provided = request.headers.get("X-API-Key")
    if not provided:
        auth = request.headers.get("Authorization", "")
        if auth.startswith("Bearer "):
            provided = auth[7:]
    if provided and provided != QUANTX_API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid QuantX API Key. Access unauthorized."
        )
    return True


# =========================================================
# UTILITY FUNCTIONS FOR SAFE CALCULATIONS
# =========================================================

def safe_float(val: Any, default: float = 0.0) -> float:
    """Safely converts values to standard float, preventing NaN, Inf, or JSON serialization errors."""
    try:
        if val is None or pd.isna(val) or math.isnan(val) or math.isinf(val):
            return default
        return float(val)
    except (TypeError, ValueError):
        return default


# =========================================================
# GLOBAL VALIDATION EXCEPTION HANDLER
# =========================================================

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Formats Pydantic validation errors into clean, JSON-serializable messages for the UI."""
    error_details = []
    clean_errors = []
    for err in exc.errors():
        loc_parts = [str(l) for l in err.get("loc", []) if l != "body"]
        field_name = " -> ".join(loc_parts)
        msg = err.get("msg", "Invalid value")
        if msg.startswith("Value error, "):
            msg = msg[len("Value error, "):]

        if field_name:
            error_details.append(f"{field_name}: {msg}")
        else:
            error_details.append(msg)

        clean_errors.append({
            "field": field_name,
            "message": msg,
            "type": err.get("type", "value_error")
        })

    joined_error = "; ".join(error_details) if error_details else "Request validation failed"
    return JSONResponse(
        status_code=422,
        content={
            "status": "validation_error",
            "message": joined_error,
            "detail": joined_error,
            "errors": clean_errors
        }
    )


# =========================================================
# REQUEST MODEL WITH STRICT VALIDATION
# =========================================================

class BacktestRequest(BaseModel):
    asset: str = Field(default="gold", description="Asset identifier ('gold', 'bitcoin', 'nvidia')")
    period: str = Field(default="5y", description="Time period ('1y', '3y', '5y', '10y')")
    strategy: str = Field(default="sma", description="Trading strategy ('sma', 'ema', 'momentum', 'mean_reversion')")
    initial_capital: float = Field(
        default=100000.0,
        gt=0,
        le=1_000_000_000.0,
        description="Initial capital (₹1,000 to ₹1,000,000,000)"
    )
    transaction_cost: float = Field(
        default=0.001,
        ge=0.0,
        le=0.10,
        description="Transaction fee rate (0.0 to 0.10, max 10%)"
    )
    short_window: int = Field(
        default=20,
        ge=2,
        le=500,
        description="Fast / short moving average window (2 to 500)"
    )
    long_window: int = Field(
        default=50,
        ge=3,
        le=1000,
        description="Slow / long moving average window (3 to 1000)"
    )

    @field_validator("asset")
    @classmethod
    def validate_asset(cls, v: str) -> str:
        clean = v.strip().lower()
        if clean not in ALLOWED_ASSETS:
            raise ValueError(f"Invalid asset '{v}'. Allowed assets: {', '.join(sorted(ALLOWED_ASSETS))}")
        return clean

    @field_validator("period")
    @classmethod
    def validate_period(cls, v: str) -> str:
        clean = v.strip().lower()
        if clean not in ALLOWED_PERIODS:
            raise ValueError(f"Invalid period '{v}'. Allowed periods: {', '.join(sorted(ALLOWED_PERIODS))}")
        return clean

    @field_validator("strategy")
    @classmethod
    def validate_strategy(cls, v: str) -> str:
        clean = v.strip().lower()
        if clean not in ALLOWED_STRATEGIES:
            raise ValueError(f"Invalid strategy '{v}'. Allowed strategies: {', '.join(sorted(ALLOWED_STRATEGIES))}")
        return clean

    @field_validator("initial_capital")
    @classmethod
    def validate_capital_min(cls, v: float) -> float:
        if v < 1000:
            raise ValueError("Initial capital must be at least Rs. 1,000 ($1,000)")
        return v

    @model_validator(mode="after")
    def validate_windows(self):
        if self.short_window >= self.long_window:
            raise ValueError(
                f"Short window ({self.short_window}) must be strictly less than long window ({self.long_window})"
            )
        return self


# =========================================================
# ROOT & DASHBOARD
# =========================================================

@app.api_route("/", methods=["GET", "HEAD"], include_in_schema=False)
@app.api_route("/login", methods=["GET", "HEAD"], include_in_schema=False)
@app.api_route("/landing", methods=["GET", "HEAD"], include_in_schema=False)
def login_view():
    login_path = os.path.join(os.path.dirname(__file__), "login.html")
    if os.path.exists(login_path):
        return FileResponse(login_path)
    frontend_path = os.path.join(os.path.dirname(__file__), "frontend.html")
    if os.path.exists(frontend_path):
        return FileResponse(frontend_path)
    return {
        "message": "QuantPulse Research Terminal",
        "status": "running"
    }


@app.api_route("/terminal", methods=["GET", "HEAD"], include_in_schema=False)
@app.api_route("/app", methods=["GET", "HEAD"], include_in_schema=False)
@app.api_route("/dashboard-desk", methods=["GET", "HEAD"], include_in_schema=False)
def terminal_view():
    frontend_path = os.path.join(os.path.dirname(__file__), "frontend.html")
    if os.path.exists(frontend_path):
        return FileResponse(frontend_path)
    return {
        "message": "QuantPulse Terminal Desk",
        "status": "running"
    }


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return Response(status_code=204)


@app.get("/api")
def api_info():
    return {
        "message": "QuantX Financial Intelligence API",
        "status": "running",
        "version": "1.1",
        "api_key_configured": True,
        "api_key_preview": f"{QUANTX_API_KEY[:8]}...{QUANTX_API_KEY[-4:]}",
        "supported_assets": list(ASSETS.keys()),
        "supported_strategies": list(ALLOWED_STRATEGIES),
        "supported_periods": list(ALLOWED_PERIODS)
    }


@app.get("/api-key/status")
@app.get("/auth/key")
def get_api_key_status(request: Request):
    provided = request.headers.get("X-API-Key") or request.query_params.get("api_key") or QUANTX_API_KEY
    is_valid = (provided == QUANTX_API_KEY)
    masked = f"{QUANTX_API_KEY[:8]}...{QUANTX_API_KEY[-4:]}"
    return {
        "status": "authenticated" if is_valid else "unauthorized",
        "valid": is_valid,
        "api_key_configured": True,
        "key_preview": masked,
        "key_prefix": QUANTX_API_KEY[:8],
        "tier": "Enterprise Quantitative Intelligence",
        "permissions": ["backtest", "market_analysis", "risk_analysis", "correlation"],
        "rate_limit": "unlimited"
    }


# =========================================================
# ASSETS
# =========================================================

@app.get("/assets")
def get_assets():
    return {
        "assets": [
            {"name": "Gold", "symbol": "GC=F", "id": "gold"},
            {"name": "Bitcoin", "symbol": "BTC-USD", "id": "bitcoin"},
            {"name": "NVIDIA", "symbol": "NVDA", "id": "nvidia"}
        ]
    }


# =========================================================
# DOWNLOAD DATA WITH ERROR HANDLING
# =========================================================

def get_data(asset: str, period: str = "5y"):
    clean_asset = str(asset).strip().lower()
    if clean_asset not in ASSETS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid asset '{asset}'. Supported assets: {', '.join(sorted(ASSETS.keys()))}"
        )

    clean_period = str(period).strip().lower()
    if clean_period not in ALLOWED_PERIODS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid period '{period}'. Supported periods: {', '.join(sorted(ALLOWED_PERIODS))}"
        )

    symbol = ASSETS[clean_asset]
    try:
        data = yf.download(
            symbol,
            period=clean_period,
            auto_adjust=True,
            progress=False
        )
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Failed to fetch market data from upstream provider for {symbol}: {str(exc)}"
        )

    if data is None or data.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No historical price data returned for '{asset}' ({symbol}) over period '{period}'"
        )

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col not in data.columns:
            raise HTTPException(
                status_code=500,
                detail=f"Missing expected price column '{col}' in downloaded data for {symbol}"
            )

    data = data[["Open", "High", "Low", "Close", "Volume"]].dropna()

    if len(data) < 10:
        raise HTTPException(
            status_code=400,
            detail=f"Insufficient historical data records ({len(data)}) for analysis"
        )

    return data


# =========================================================
# INDICATORS
# =========================================================

def add_indicators(data: pd.DataFrame, short_window: int = 20, long_window: int = 50) -> pd.DataFrame:
    data = data.copy()

    # SMA
    data["SMA_SHORT"] = data["Close"].rolling(short_window).mean()
    data["SMA_LONG"] = data["Close"].rolling(long_window).mean()

    # EMA
    data["EMA_SHORT"] = data["Close"].ewm(span=short_window, adjust=False).mean()
    data["EMA_LONG"] = data["Close"].ewm(span=long_window, adjust=False).mean()

    # Daily returns
    data["Returns"] = data["Close"].pct_change()

    # Cumulative returns
    data["Cumulative_Return"] = (1 + data["Returns"].fillna(0)).cumprod() - 1

    return data


# =========================================================
# STRATEGY SIGNALS
# =========================================================

def create_strategy(data: pd.DataFrame, strategy: str, short_window: int, long_window: int) -> pd.DataFrame:
    data = data.copy()
    data["Signal"] = 0

    clean_strategy = strategy.strip().lower()

    if clean_strategy == "sma":
        data.loc[data["SMA_SHORT"] > data["SMA_LONG"], "Signal"] = 1
    elif clean_strategy == "ema":
        data.loc[data["EMA_SHORT"] > data["EMA_LONG"], "Signal"] = 1
    elif clean_strategy == "momentum":
        momentum = data["Close"].pct_change(20)
        data.loc[momentum > 0, "Signal"] = 1
    elif clean_strategy == "mean_reversion":
        mean = data["Close"].rolling(20).mean()
        data.loc[data["Close"] < mean, "Signal"] = 1
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid strategy '{strategy}'. Supported strategies: {', '.join(sorted(ALLOWED_STRATEGIES))}"
        )

    return data


# =========================================================
# BACKTEST ENGINE
# =========================================================

def run_backtest(data: pd.DataFrame, initial_capital: float, transaction_cost: float, asset_name: str = "ASSET"):
    data = data.copy()
    capital = float(initial_capital)
    position = 0.0
    portfolio_values = []
    trades = 0
    trade_records = []

    for i in range(len(data)):
        price = safe_float(data["Close"].iloc[i], 1.0)
        signal = int(data["Signal"].iloc[i]) if pd.notna(data["Signal"].iloc[i]) else 0
        idx = data.index[i]
        date_str = idx.strftime("%Y-%m-%d") if hasattr(idx, 'strftime') else str(idx)

        # BUY
        if signal == 1 and position == 0:
            if price > 0:
                position = capital / price
                capital = 0.0
                trades += 1
                trade_records.append({
                    "date": date_str,
                    "asset": asset_name.upper(),
                    "action": "BUY",
                    "price": round(price, 2),
                    "shares": round(position, 4),
                    "cost": 0.0,
                    "portfolio_value": round(position * price, 2)
                })

        # SELL
        elif signal == 0 and position > 0:
            gross = position * price
            cost = gross * transaction_cost
            capital = gross - cost
            trades += 1
            trade_records.append({
                "date": date_str,
                "asset": asset_name.upper(),
                "action": "SELL",
                "price": round(price, 2),
                "shares": round(position, 4),
                "cost": round(cost, 2),
                "portfolio_value": round(capital, 2)
            })
            position = 0.0

        # PORTFOLIO VALUE
        if position > 0:
            value = position * price
        else:
            value = capital
        portfolio_values.append(value)

    # Close open position at end
    if position > 0:
        final_price = safe_float(data["Close"].iloc[-1], 1.0)
        last_idx = data.index[-1]
        last_date = last_idx.strftime("%Y-%m-%d") if hasattr(last_idx, 'strftime') else str(last_idx)
        gross = position * final_price
        cost = gross * transaction_cost
        capital = gross - cost
        trade_records.append({
            "date": last_date,
            "asset": asset_name.upper(),
            "action": "CLOSE",
            "price": round(final_price, 2),
            "shares": round(position, 4),
            "cost": round(cost, 2),
            "portfolio_value": round(capital, 2)
        })

    data["Portfolio"] = portfolio_values
    final_value = capital
    return data, final_value, trades, trade_records


# =========================================================
# PERFORMANCE METRICS
# =========================================================

def calculate_metrics(data: pd.DataFrame, initial_capital: float, final_value: float) -> Dict[str, float]:
    portfolio = data["Portfolio"]
    if initial_capital > 0:
        total_return = (final_value / initial_capital) - 1
    else:
        total_return = 0.0

    daily_returns = portfolio.pct_change().dropna()

    if len(daily_returns) > 1:
        vol_std = daily_returns.std()
        volatility = vol_std * np.sqrt(252) if (vol_std > 0 and not np.isnan(vol_std)) else 0.0
    else:
        volatility = 0.0

    if volatility > 0:
        annual_return = daily_returns.mean() * 252
        sharpe = annual_return / volatility if not np.isnan(annual_return) else 0.0
    else:
        sharpe = 0.0

    rolling_max = portfolio.cummax()
    drawdown = (portfolio / rolling_max) - 1
    max_drawdown = drawdown.min() if len(drawdown) > 0 else 0.0

    return {
        "total_return": round(safe_float(total_return) * 100, 2),
        "volatility": round(safe_float(volatility) * 100, 2),
        "sharpe_ratio": round(safe_float(sharpe), 2),
        "max_drawdown": round(safe_float(max_drawdown) * 100, 2),
        "final_value": round(safe_float(final_value), 2)
    }


# =========================================================
# BUY AND HOLD BENCHMARK
# =========================================================

def buy_and_hold(data: pd.DataFrame, initial_capital: float) -> Dict[str, float]:
    first_price = safe_float(data["Close"].iloc[0], 1.0)
    last_price = safe_float(data["Close"].iloc[-1], 1.0)

    if first_price > 0:
        final_value = initial_capital * (last_price / first_price)
    else:
        final_value = initial_capital

    if initial_capital > 0:
        total_return = (final_value / initial_capital) - 1
    else:
        total_return = 0.0

    bh_daily_returns = data["Close"].pct_change().dropna()
    if len(bh_daily_returns) > 1:
        bh_std = bh_daily_returns.std()
        volatility = bh_std * np.sqrt(252) if (bh_std > 0 and not np.isnan(bh_std)) else 0.0
    else:
        volatility = 0.0

    if volatility > 0:
        annual_return = bh_daily_returns.mean() * 252
        sharpe = annual_return / volatility if not np.isnan(annual_return) else 0.0
    else:
        sharpe = 0.0

    return {
        "final_value": round(safe_float(final_value), 2),
        "return": round(safe_float(total_return) * 100, 2),
        "sharpe_ratio": round(safe_float(sharpe), 2)
    }


# =========================================================
# BACKTEST API
# =========================================================

@app.post("/backtest")
def backtest(request: BacktestRequest, raw_req: Request):
    check_api_key(raw_req)
    # 1. Download market data with validation
    data = get_data(request.asset, request.period)

    # 2. Add indicators
    data = add_indicators(data, request.short_window, request.long_window)

    # 3. Apply strategy signals
    data = create_strategy(data, request.strategy, request.short_window, request.long_window)

    # 4. Filter valid rows
    data = data.dropna()

    if len(data) < 5:
        raise HTTPException(
            status_code=400,
            detail=f"After indicator calculation, insufficient data rows ({len(data)}) remained. Try a longer period."
        )

    # 5. Run backtest simulation
    result_data, final_value, trades, trade_records = run_backtest(
        data,
        request.initial_capital,
        request.transaction_cost,
        request.asset
    )

    # 6. Calculate strategy metrics
    metrics = calculate_metrics(result_data, request.initial_capital, final_value)

    # 7. Calculate Buy & Hold benchmark
    benchmark = buy_and_hold(result_data, request.initial_capital)

    # 8. Build chart data safely
    chart_data = []
    for index, row in result_data.iterrows():
        chart_data.append({
            "date": index.strftime("%Y-%m-%d") if hasattr(index, 'strftime') else str(index),
            "price": round(safe_float(row["Close"]), 4),
            "sma_short": round(safe_float(row["SMA_SHORT"]), 4),
            "sma_long": round(safe_float(row["SMA_LONG"]), 4),
            "ema_short": round(safe_float(row["EMA_SHORT"]), 4),
            "ema_long": round(safe_float(row["EMA_LONG"]), 4),
            "portfolio": round(safe_float(row["Portfolio"]), 2),
            "signal": int(row["Signal"]) if pd.notna(row["Signal"]) else 0
        })

    return {
        "asset": request.asset,
        "strategy": request.strategy,
        "period": request.period,
        "metrics": metrics,
        "trades": trades,
        "trades_history": trade_records,
        "buy_and_hold": benchmark,
        "chart": chart_data
    }


# =========================================================
# ASSET ANALYSIS
# =========================================================

@app.get("/analysis/{asset}")
def analysis(asset: str, raw_req: Request):
    check_api_key(raw_req)
    clean_asset = asset.strip().lower()
    if clean_asset not in ASSETS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid asset '{asset}'. Supported assets: {', '.join(sorted(ASSETS.keys()))}"
        )

    data = get_data(clean_asset, "5y")
    data = add_indicators(data)
    returns = data["Returns"].dropna()

    if len(returns) > 1 and returns.std() > 0:
        volatility = returns.std() * np.sqrt(252)
        annual_return = returns.mean() * 252
        sharpe = annual_return / volatility if volatility > 0 else 0.0
    else:
        volatility = 0.0
        sharpe = 0.0

    cumulative = (1 + returns.fillna(0)).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative / peak) - 1

    return {
        "asset": clean_asset,
        "latest_price": round(safe_float(data["Close"].iloc[-1]), 2),
        "daily_return": round(safe_float(data["Returns"].iloc[-1]) * 100, 2),
        "volatility": round(safe_float(volatility) * 100, 2),
        "sharpe_ratio": round(safe_float(sharpe), 2),
        "max_drawdown": round(safe_float(drawdown.min()) * 100, 2)
    }


# =========================================================
# CORRELATION MATRIX
# =========================================================

@app.get("/correlation")
def correlation(raw_req: Request):
    check_api_key(raw_req)
    frames = {}
    for asset, symbol in ASSETS.items():
        try:
            data = yf.download(symbol, period="5y", auto_adjust=True, progress=False)
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            frames[asset] = data["Close"].pct_change()
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Failed to fetch data for {asset}: {str(e)}")

    returns = pd.DataFrame(frames).dropna()
    matrix = returns.corr().round(2).to_dict()

    return {
        "correlation": matrix
    }


# =========================================================
# MARKET ANALYSIS
# =========================================================

@app.get("/market-analysis/{asset}")
def market_analysis(asset: str, raw_req: Request, period: str = "5y"):
    check_api_key(raw_req)
    clean_asset = asset.strip().lower()
    clean_period = period.strip().lower()

    if clean_asset not in ASSETS:
        raise HTTPException(status_code=400, detail=f"Invalid asset '{asset}'. Supported: {', '.join(sorted(ASSETS.keys()))}")
    if clean_period not in ALLOWED_PERIODS:
        raise HTTPException(status_code=400, detail=f"Invalid period '{period}'. Supported: {', '.join(sorted(ALLOWED_PERIODS))}")

    data = get_data(clean_asset, clean_period)
    data = add_indicators(data, short_window=20, long_window=50)

    data["Rolling_Vol"] = data["Returns"].rolling(20).std() * np.sqrt(252) * 100

    latest_price = safe_float(data["Close"].iloc[-1])
    high_52w = safe_float(data["High"].tail(252).max()) if len(data) >= 252 else safe_float(data["High"].max())
    low_52w = safe_float(data["Low"].tail(252).min()) if len(data) >= 252 else safe_float(data["Low"].min())

    first_close = safe_float(data["Close"].iloc[0], 1.0)
    total_return = ((latest_price / first_close) - 1) * 100 if first_close > 0 else 0.0
    annualized_vol = safe_float(data["Returns"].std() * np.sqrt(252) * 100)

    timeseries = []
    for index, row in data.iterrows():
        timeseries.append({
            "date": index.strftime("%Y-%m-%d") if hasattr(index, 'strftime') else str(index),
            "price": round(safe_float(row["Close"]), 2),
            "sma20": round(safe_float(row["SMA_SHORT"]), 2) if pd.notna(row["SMA_SHORT"]) else None,
            "sma50": round(safe_float(row["SMA_LONG"]), 2) if pd.notna(row["SMA_LONG"]) else None,
            "ema20": round(safe_float(row["EMA_SHORT"]), 2) if pd.notna(row["EMA_SHORT"]) else None,
            "ema50": round(safe_float(row["EMA_LONG"]), 2) if pd.notna(row["EMA_LONG"]) else None,
            "daily_return": round(safe_float(row["Returns"]) * 100, 2) if pd.notna(row["Returns"]) else 0.0,
            "cumulative_return": round(safe_float(row["Cumulative_Return"]) * 100, 2) if pd.notna(row["Cumulative_Return"]) else 0.0,
            "rolling_vol": round(safe_float(row["Rolling_Vol"]), 2) if pd.notna(row["Rolling_Vol"]) else None,
        })

    return {
        "asset": clean_asset,
        "period": clean_period,
        "summary": {
            "latest_price": round(latest_price, 2),
            "high_52w": round(high_52w, 2),
            "low_52w": round(low_52w, 2),
            "total_return": round(safe_float(total_return), 2),
            "annualized_vol": round(annualized_vol, 2)
        },
        "timeseries": timeseries
    }


# =========================================================
# RISK ANALYSIS
# =========================================================

@app.get("/risk-analysis/{asset}")
def risk_analysis(asset: str, raw_req: Request, period: str = "5y"):
    check_api_key(raw_req)
    clean_asset = asset.strip().lower()
    clean_period = period.strip().lower()

    if clean_asset not in ASSETS:
        raise HTTPException(status_code=400, detail=f"Invalid asset '{asset}'. Supported: {', '.join(sorted(ASSETS.keys()))}")
    if clean_period not in ALLOWED_PERIODS:
        raise HTTPException(status_code=400, detail=f"Invalid period '{period}'. Supported: {', '.join(sorted(ALLOWED_PERIODS))}")

    data = get_data(clean_asset, clean_period)
    returns = data["Close"].pct_change().dropna()

    if len(returns) < 5:
        raise HTTPException(status_code=400, detail="Insufficient return data points for risk analysis")

    ret_std = returns.std()
    ret_mean = returns.mean()

    daily_vol = float(ret_std * 100) if not np.isnan(ret_std) else 0.0
    annualized_vol = float(ret_std * np.sqrt(252) * 100) if not np.isnan(ret_std) else 0.0
    annualized_return = float(ret_mean * 252 * 100) if not np.isnan(ret_mean) else 0.0

    sharpe = float(annualized_return / annualized_vol) if annualized_vol > 0 else 0.0

    downside = returns[returns < 0]
    downside_std = downside.std() if len(downside) > 0 else 0.0
    downside_dev = float(downside_std * np.sqrt(252) * 100) if not np.isnan(downside_std) else 0.0
    sortino = float(annualized_return / downside_dev) if downside_dev > 0 else 0.0

    cumulative = (1 + returns).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative / peak) - 1
    max_drawdown = float(drawdown.min() * 100) if len(drawdown) > 0 else 0.0

    rolling_vol = returns.rolling(30).std() * np.sqrt(252) * 100

    var_95_hist = float(np.percentile(returns, 5) * 100)
    var_95_param = float((ret_mean - 1.645 * ret_std) * 100)

    timeseries = []
    for idx in returns.index:
        date_str = idx.strftime("%Y-%m-%d") if hasattr(idx, 'strftime') else str(idx)
        dd_val = round(safe_float(drawdown.loc[idx]) * 100, 2) if idx in drawdown.index else 0.0
        rvol_val = round(safe_float(rolling_vol.loc[idx]), 2) if (idx in rolling_vol.index and pd.notna(rolling_vol.loc[idx])) else None
        ret_val = round(safe_float(returns.loc[idx]) * 100, 2)

        timeseries.append({
            "date": date_str,
            "drawdown": dd_val,
            "rolling_vol": rvol_val,
            "return": ret_val
        })

    return {
        "asset": clean_asset,
        "period": clean_period,
        "metrics": {
            "daily_volatility": round(safe_float(daily_vol), 2),
            "annualized_volatility": round(safe_float(annualized_vol), 2),
            "sharpe_ratio": round(safe_float(sharpe), 2),
            "sortino_ratio": round(safe_float(sortino), 2),
            "max_drawdown": round(safe_float(max_drawdown), 2),
            "var_95_daily": round(safe_float(var_95_hist), 2),
            "var_95_parametric": round(safe_float(var_95_param), 2)
        },
        "timeseries": timeseries
    }


# =========================================================
# ROLLING CORRELATION
# =========================================================

@app.get("/correlation/rolling")
def rolling_correlation(
    raw_req: Request,
    window: int = Query(default=60, ge=5, le=500, description="Rolling window in days (5 to 500)"),
    period: str = "5y"
):
    check_api_key(raw_req)
    clean_period = period.strip().lower()
    if clean_period not in ALLOWED_PERIODS:
        raise HTTPException(status_code=400, detail=f"Invalid period '{period}'. Supported: {', '.join(sorted(ALLOWED_PERIODS))}")

    frames = {}
    for asset, symbol in ASSETS.items():
        try:
            d = yf.download(symbol, period=clean_period, auto_adjust=True, progress=False)
            if isinstance(d.columns, pd.MultiIndex):
                d.columns = d.columns.get_level_values(0)
            frames[asset] = d["Close"].pct_change()
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Failed to fetch data for {asset}: {str(e)}")

    returns = pd.DataFrame(frames).dropna()

    if len(returns) <= window:
        raise HTTPException(status_code=400, detail=f"Rolling window ({window}) exceeds available data points ({len(returns)})")

    roll_gb = returns["gold"].rolling(window).corr(returns["bitcoin"])
    roll_gn = returns["gold"].rolling(window).corr(returns["nvidia"])
    roll_bn = returns["bitcoin"].rolling(window).corr(returns["nvidia"])

    combined = pd.DataFrame({
        "gold_bitcoin": roll_gb,
        "gold_nvidia": roll_gn,
        "bitcoin_nvidia": roll_bn
    }).dropna()

    timeseries = []
    for idx, row in combined.iterrows():
        timeseries.append({
            "date": idx.strftime("%Y-%m-%d") if hasattr(idx, 'strftime') else str(idx),
            "gold_bitcoin": round(safe_float(row["gold_bitcoin"]), 2),
            "gold_nvidia": round(safe_float(row["gold_nvidia"]), 2),
            "bitcoin_nvidia": round(safe_float(row["bitcoin_nvidia"]), 2)
        })

    return {
        "window": window,
        "period": clean_period,
        "timeseries": timeseries
    }


# =========================================================
# HEALTH CHECK
# =========================================================

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "validation": "strict",
        "version": "1.1"
    }


# =========================================================
# RUNNER
# =========================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)