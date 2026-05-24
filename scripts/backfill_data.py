"""QuantForge historical data backfill — parallel fetch 2y daily quotes for all stocks.

Usage: python scripts/backfill_data.py [--workers 8] [--start 20240101] [--batch 500]
"""
import argparse
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta

sys.path.insert(0, ".")

from app.core.database import SessionLocal, engine
from app.models.stock import Stock
from app.models.market import DailyQuote
from app.services.data_pipeline import DataPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("backfill")


def fetch_one_stock(code: str, start_date: str, end_date: str) -> tuple[str, dict | None, str | None]:
    """Fetch single stock data. Returns (code, data_dict, error)."""
    try:
        data = DataPipeline.fetch_daily_quotes([code], start_date, end_date)
        if data:
            return code, data, None
        return code, None, "no data returned"
    except Exception as e:
        return code, None, str(e)


def save_batch(data: dict[str, dict], db_session):
    """Save a batch of stock data to DB."""
    try:
        saved = DataPipeline.save_daily_quotes(data, db_session)
        return saved
    except Exception as e:
        logger.error(f"Save batch failed: {e}")
        return 0


def main():
    parser = argparse.ArgumentParser(description="Backfill historical daily quotes")
    parser.add_argument("--workers", type=int, default=8, help="Parallel workers")
    parser.add_argument("--start", type=str, default="20240101", help="Start date YYYYMMDD")
    parser.add_argument("--batch", type=int, default=500, help="Batch size for saving")
    args = parser.parse_args()

    end_date = date.today().strftime("%Y%m%d")
    start_date = args.start

    logger.info(f"Backfill: {start_date} ~ {end_date}, {args.workers} workers")

    # Get all active stock codes
    db = SessionLocal()
    codes = [r[0] for r in db.query(Stock.code).filter(Stock.is_active == True).all()]
    db.close()

    logger.info(f"Target: {len(codes)} stocks")

    total_saved = 0
    success = 0
    failed = 0

    # Process in batches for memory control
    batch_size = args.batch
    for batch_start in range(0, len(codes), batch_size):
        batch_codes = codes[batch_start : batch_start + batch_size]
        logger.info(f"Batch {batch_start // batch_size + 1}/{(len(codes) - 1) // batch_size + 1}: "
                    f"{len(batch_codes)} stocks")

        batch_data = {}
        batch_start_time = time.time()

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(fetch_one_stock, code, start_date, end_date): code
                for code in batch_codes
            }

            for future in as_completed(futures):
                code = futures[future]
                try:
                    code_result, data, error = future.result()
                    if data:
                        batch_data.update(data)
                        success += 1
                    else:
                        failed += 1
                        if failed <= 5:
                            logger.warning(f"Failed {code}: {error}")
                except Exception as e:
                    failed += 1
                    if failed <= 5:
                        logger.warning(f"Future error for {code}: {e}")

        # Save batch
        if batch_data:
            db = SessionLocal()
            try:
                saved = DataPipeline.save_daily_quotes(batch_data, db)
                total_saved += saved
                logger.info(f"  Saved {saved} quotes. "
                            f"Total: {total_saved} quotes, {success} stocks OK, {failed} failed")
            finally:
                db.close()

        elapsed = time.time() - batch_start_time
        logger.info(f"  Batch completed in {elapsed:.1f}s ({elapsed / len(batch_codes):.2f}s per stock)")

        # Brief pause to avoid rate limiting
        time.sleep(1)

    logger.info(f"Backfill complete: {total_saved} total quotes, {success} stocks OK, {failed} failed")


if __name__ == "__main__":
    main()
