# QuantX Frontend API Integration & Validation Reference

This document describes the validation rules and API integration implemented between `frontend.html` and the FastAPI backend (`http://127.0.0.1:8000`).

---

## 1. Validation Rules Matrix

| Parameter | Validation Rule | Constraints | UI Feedback |
| :--- | :--- | :--- | :--- |
| **`asset`** | Must be a recognized ticker identifier | `"gold"`, `"bitcoin"`, `"nvidia"` | Dropdown selection |
| **`strategy`** | Must be an implemented quantitative algorithm | `"sma"`, `"ema"`, `"momentum"`, `"mean_reversion"` | Dropdown selection |
| **`period`** | Valid historical time horizon | `"1mo"`, `"3mo"`, `"6mo"`, `"1y"`, `"2y"`, `"3y"`, `"5y"`, `"10y"`, `"ytd"`, `"max"` | Dropdown selection |
| **`initial_capital`** | Positive floating-point value | Min: **₹1,000**, Max: **₹1,000,000,000** | Glowing red border + error span + toast alert |
| **`transaction_cost`** | Percentage fee rate | Range: **0.00% to 10.00%** (`0.0` to `0.10`) | Glowing red border + error span + toast alert |
| **`short_window`** | Fast moving average integer window | Min: **2**, Max: **500**, **`short_window < long_window`** | Glowing red border + cross-field error message |
| **`long_window`** | Slow moving average integer window | Min: **3**, Max: **1,000**, **`long_window > short_window`** | Glowing red border + cross-field error message |

---

## 2. Front-end Validation Implementation

```javascript
// Validates all parameter inputs before sending HTTP request
function validateAllBtInputs() {
    const capOk = validateBtCapital();
    const costOk = validateBtCost();
    const winOk = validateBtWindows();
    return capOk && costOk && winOk;
}

// Submits validated payload and handles errors gracefully
async function runBacktest() {
    if (!validateAllBtInputs()) {
        showToast("Validation Error", "Please resolve highlighted parameters before running backtest", "warning");
        return;
    }

    const asset = document.getElementById("asset").value;
    const strategy = document.getElementById("strategy").value;
    const capital = Number(document.getElementById("capital").value);
    const cost = Number(document.getElementById("transaction").value) / 100;
    const shortW = Number(document.getElementById("shortWindow").value);
    const longW = Number(document.getElementById("longWindow").value);
    const period = document.getElementById("period").value;

    try {
        const response = await fetch("http://127.0.0.1:8000/backtest", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                asset: asset,
                period: period,
                strategy: strategy,
                initial_capital: capital,
                transaction_cost: cost,
                short_window: shortW,
                long_window: longW
            })
        });

        if (!response.ok) {
            const errPayload = await response.json().catch(() => ({ detail: `HTTP ${response.status}` }));
            const msg = errPayload.detail || errPayload.message || "Backtest computation failed";
            showToast("Validation Error", msg, "error");
            throw new Error(msg);
        }

        const result = await response.json();
        showToast("Backtest Complete", `Successfully simulated ${asset.toUpperCase()}`, "success");

        // Update Scorecard & Visualizations
        document.getElementById("return").innerText = (result.metrics.total_return >= 0 ? "+" : "") + result.metrics.total_return + "%";
        document.getElementById("sharpe").innerText = result.metrics.sharpe_ratio;
        document.getElementById("volatility").innerText = result.metrics.volatility + "%";
        document.getElementById("drawdown").innerText = result.metrics.max_drawdown + "%";
        document.getElementById("trades").innerText = result.trades;
        document.getElementById("finalCapital").innerText = "₹" + Math.round(result.metrics.final_value).toLocaleString("en-IN");

    } catch (err) {
        console.error("Backtest error:", err);
    }
}
```

---

## 3. Back-end Validation Schema (`Pydantic v2`)

```python
class BacktestRequest(BaseModel):
    asset: str = Field(default="gold")
    period: str = Field(default="5y")
    strategy: str = Field(default="sma")
    initial_capital: float = Field(default=100000.0, gt=0, le=1_000_000_000.0)
    transaction_cost: float = Field(default=0.001, ge=0.0, le=0.10)
    short_window: int = Field(default=20, ge=2, le=500)
    long_window: int = Field(default=50, ge=3, le=1000)

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
```