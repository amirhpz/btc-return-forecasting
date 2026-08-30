from __future__ import annotations

REQUIRED_OHLCV_COLUMNS: tuple[str, ...] = (
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
)

PRICE_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close")
ADDITIVE_COLUMNS: tuple[str, ...] = (
    "volume",
    "quote_asset_volume",
    "number_of_trades",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
)
