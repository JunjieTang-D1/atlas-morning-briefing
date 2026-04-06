#!/usr/bin/env python3
# Copyright (c) 2026 Junjie Tang. MIT License. See LICENSE file for details.
"""
Stock data fetcher.

Fetches stock market data using Finnhub API.
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import requests

from scripts.utils import load_config


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class StockFetcher:
    """Fetches stock market data from Finnhub API."""

    FINNHUB_API_URL = "https://finnhub.io/api/v1"

    def __init__(self, api_key: str, symbols: List[str], request_delay: float = 0.5):
        """
        Initialize StockFetcher.

        Args:
            api_key: Finnhub API key
            symbols: List of stock symbols to fetch
            request_delay: Seconds to wait between API calls
        """
        self.api_key = api_key
        self.symbols = symbols
        self.request_delay = request_delay

    def fetch_quote(self, symbol: str) -> Dict[str, Any]:
        """
        Fetch current quote for a stock symbol.

        Args:
            symbol: Stock symbol (e.g., 'AAPL')

        Returns:
            Quote data dictionary
        """
        try:
            logger.info(f"Fetching quote for {symbol}")
            url = f"{self.FINNHUB_API_URL}/quote"
            headers = {"X-Finnhub-Token": self.api_key}
            params = {"symbol": symbol}

            response = requests.get(url, headers=headers, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            quote = {
                "symbol": symbol,
                "current_price": data.get("c", 0),
                "change": data.get("d", 0),
                "percent_change": data.get("dp", 0),
                "high": data.get("h", 0),
                "low": data.get("l", 0),
                "open": data.get("o", 0),
                "previous_close": data.get("pc", 0),
                "timestamp": data.get("t", 0),
            }

            return quote

        except requests.RequestException as e:
            logger.error(f"Failed to fetch quote for {symbol}: {e}")
            return {"symbol": symbol, "error": str(e)}

    def fetch_company_profile(self, symbol: str) -> Dict[str, Any]:
        """
        Fetch company profile for a stock symbol.

        Args:
            symbol: Stock symbol

        Returns:
            Company profile dictionary
        """
        try:
            logger.debug(f"Fetching company profile for {symbol}")
            url = f"{self.FINNHUB_API_URL}/stock/profile2"
            headers = {"X-Finnhub-Token": self.api_key}
            params = {"symbol": symbol}

            response = requests.get(url, headers=headers, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            profile = {
                "name": data.get("name", symbol),
                "ticker": data.get("ticker", symbol),
                "exchange": data.get("exchange", ""),
                "industry": data.get("finnhubIndustry", ""),
                "market_cap": data.get("marketCapitalization", 0),
                "currency": data.get("currency", "USD"),
            }

            return profile

        except requests.RequestException as e:
            logger.warning(f"Failed to fetch profile for {symbol}: {e}")
            return {"name": symbol, "ticker": symbol}

    def fetch_all_stocks(self) -> List[Dict[str, Any]]:
        """
        Fetch data for all configured stocks.

        Returns:
            List of stock data dictionaries
        """
        all_stocks = []

        for i, symbol in enumerate(self.symbols):
            # Rate limit between requests
            if i > 0 and self.request_delay > 0:
                time.sleep(self.request_delay)

            # Fetch quote
            quote = self.fetch_quote(symbol)

            # Fetch company profile if quote succeeded
            if "error" not in quote:
                if self.request_delay > 0:
                    time.sleep(self.request_delay)
                profile = self.fetch_company_profile(symbol)
                stock_data = {**quote, **profile}
            else:
                stock_data = quote

            all_stocks.append(stock_data)

        logger.info(f"Fetched data for {len(all_stocks)} stocks")
        return all_stocks



def main() -> int:
    """
    Main entry point for stock_fetcher.

    Returns:
        Exit code (0 for success, 1 for partial failure, 2 for total failure)
    """
    parser = argparse.ArgumentParser(description="Fetch stock market data")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration YAML file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="stocks.json",
        help="Output JSON file path",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    args = parser.parse_args()

    # Set log level
    logger.setLevel(getattr(logging, args.log_level))

    # Get API key from environment
    api_key = os.environ.get("FINNHUB_API_KEY")
    if not api_key:
        logger.error("FINNHUB_API_KEY environment variable not set")
        return 2

    # Load config
    config = load_config(args.config)

    # Extract settings
    symbols = config.get("stocks", [])

    if not symbols:
        logger.error("No stocks configured")
        return 2

    # Fetch stock data
    fetcher = StockFetcher(api_key=api_key, symbols=symbols)
    stocks = fetcher.fetch_all_stocks()

    if not stocks:
        logger.warning("No stock data fetched")
        return 1

    # Check for errors
    errors = [s for s in stocks if "error" in s]
    if errors:
        logger.warning(f"{len(errors)} stocks failed to fetch")
        if len(errors) == len(stocks):
            return 2  # All failed

    # Save results
    try:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(stocks, f, indent=2)
        logger.info(f"Saved data for {len(stocks)} stocks to {args.output}")
        return 0 if not errors else 1
    except IOError as e:
        logger.error(f"Failed to write output file: {e}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
