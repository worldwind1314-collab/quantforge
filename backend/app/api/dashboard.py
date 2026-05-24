"""Dashboard web interface — Jinja2 templates with ECharts visualization."""

from datetime import date, timedelta

import numpy as np
from fastapi import APIRouter, Depends, Request
from fastapi.templating import Jinja2Templates
from sqlalchemy import func
from sqlalchemy.orm import Session

from pathlib import Path

from ..core.database import get_db
from ..models.finance import FactorScore, FinancialIndicator, MLPrediction
from ..models.market import DailyQuote
from ..models.stock import Stock
from ..models.trading import BacktestResult

router = APIRouter(tags=["dashboard"])
_tmpl_dir = Path(__file__).parent.parent / "templates"
templates = Jinja2Templates(directory=str(_tmpl_dir))


def _get_stats(db: Session) -> dict:
    """Collect system-wide statistics."""
    stock_count = db.query(func.count(Stock.code)).scalar()
    quote_count = db.query(func.count(DailyQuote.id)).scalar()
    quote_codes = db.query(func.count(func.distinct(DailyQuote.code))).scalar()
    fi_count = db.query(func.count(FinancialIndicator.id)).scalar()
    fi_codes = db.query(func.count(func.distinct(FinancialIndicator.code))).scalar()
    fs_count = db.query(func.count(FactorScore.id)).scalar()
    fs_dates = db.query(func.count(func.distinct(FactorScore.trade_date))).scalar()
    ml_count = db.query(func.count(MLPrediction.id)).scalar()
    ml_dates = db.query(func.count(func.distinct(MLPrediction.trade_date))).scalar()

    latest_quote_date = db.query(func.max(DailyQuote.trade_date)).scalar()
    latest_ml_date = db.query(func.max(MLPrediction.trade_date)).scalar()

    freshness = "未知"
    if latest_quote_date:
        try:
            d = date.fromisoformat(latest_quote_date)
            diff = (date.today() - d).days
            freshness = f"{diff}天前" if diff > 0 else "今日"
        except (ValueError, TypeError):
            pass

    # Latest factor score stats
    latest_fs_date = db.query(func.max(FactorScore.trade_date)).scalar()
    composite_mean = None
    composite_std = None
    latest_codes = 0
    if latest_fs_date:
        fs_row = (
            db.query(FactorScore)
            .filter(FactorScore.trade_date == latest_fs_date)
            .all()
        )
        latest_codes = len(fs_row)
        comps = [r.composite_score for r in fs_row if r.composite_score is not None]
        if comps:
            composite_mean = float(np.mean(comps))
            composite_std = float(np.std(comps))

    return {
        "stock_count": stock_count,
        "quote_count": quote_count,
        "quote_codes": quote_codes,
        "fi_count": fi_count,
        "fi_codes": fi_codes,
        "fs_count": fs_count,
        "fs_dates": fs_dates,
        "ml_count": ml_count,
        "ml_dates": ml_dates,
        "latest_quote_date": latest_quote_date,
        "latest_ml_date": latest_ml_date,
        "freshness": freshness,
        "composite_mean": composite_mean,
        "composite_std": composite_std,
        "latest_codes": latest_codes,
    }


@router.get("/")
def dashboard_overview(request: Request, db: Session = Depends(get_db)):
    """System overview dashboard."""
    stats = _get_stats(db)

    # Top predictions
    top_predictions = []
    latest_ml_date = db.query(func.max(MLPrediction.trade_date)).scalar()
    latest_ic = None
    if latest_ml_date:
        preds = (
            db.query(MLPrediction)
            .filter(MLPrediction.trade_date == latest_ml_date)
            .order_by(MLPrediction.predicted_return.desc())
            .limit(10)
            .all()
        )
        stock_map = {}
        if preds:
            codes = [p.code for p in preds]
            stocks = db.query(Stock).filter(Stock.code.in_(codes)).all()
            stock_map = {s.code: s.name for s in stocks}

        for p in preds:
            top_predictions.append({
                "code": p.code,
                "name": stock_map.get(p.code, ""),
                "predicted_return": p.predicted_return,
                "prediction_rank": p.prediction_rank,
                "confidence": p.confidence,
            })

        # Compute latest IC from backtest
        latest_bt = (
            db.query(BacktestResult)
            .order_by(BacktestResult.created_at.desc())
            .first()
        )
        if latest_bt:
            latest_ic = latest_bt.ic_mean

    # Prediction distribution
    pred_dist = {"bins": [], "counts": []}
    if latest_ml_date:
        all_preds = (
            db.query(MLPrediction.predicted_return)
            .filter(MLPrediction.trade_date == latest_ml_date)
            .all()
        )
        vals = [p[0] for p in all_preds if p[0] is not None]
        if vals:
            hist, bins = np.histogram(vals, bins=20)
            pred_dist["bins"] = [f"{bins[i]:.2f}" for i in range(len(bins) - 1)]
            pred_dist["counts"] = hist.tolist()

    # Feature importance from latest backtest
    import json

    feature_importance = {}
    latest_bt = (
        db.query(BacktestResult)
        .filter(BacktestResult.feature_importance_json.isnot(None))
        .order_by(BacktestResult.created_at.desc())
        .first()
    )
    if latest_bt and latest_bt.feature_importance_json:
        try:
            feature_importance = json.loads(latest_bt.feature_importance_json)
        except (json.JSONDecodeError, TypeError):
            pass

    # Coverage trend (last 30 days)
    coverage_trend = {"dates": [], "quotes": [], "factors": [], "predictions": []}
    for i in range(30, 0, -1):
        d = (date.today() - timedelta(days=i)).isoformat()
        coverage_trend["dates"].append(d[-5:])
        coverage_trend["quotes"].append(
            db.query(func.count(func.distinct(DailyQuote.code)))
            .filter(DailyQuote.trade_date == d).scalar() or 0
        )
        coverage_trend["factors"].append(
            db.query(func.count(FactorScore.id))
            .filter(FactorScore.trade_date == d).scalar() or 0
        )
        coverage_trend["predictions"].append(
            db.query(func.count(MLPrediction.id))
            .filter(MLPrediction.trade_date == d).scalar() or 0
        )

    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "page": "overview",
        "stats": stats,
        "top_predictions": top_predictions,
        "pred_dist": pred_dist,
        "feature_importance": feature_importance,
        "coverage_trend": coverage_trend,
        "latest_ic": latest_ic,
    })


@router.get("/predictions")
def predictions_page(request: Request, db: Session = Depends(get_db)):
    """ML predictions ranking page."""
    stats = _get_stats(db)
    latest_date = db.query(func.max(MLPrediction.trade_date)).scalar()
    predictions = []
    total = 0
    pred_dist = {"bins": [], "counts": []}
    top_factors = []
    bot_factors = []

    if latest_date:
        preds = (
            db.query(MLPrediction)
            .filter(MLPrediction.trade_date == latest_date)
            .order_by(MLPrediction.predicted_return.desc())
            .all()
        )
        total = len(preds)

        stock_map = {}
        if preds:
            codes = list(set(p.code for p in preds))
            stocks = db.query(Stock).filter(Stock.code.in_(codes)).all()
            stock_map = {s.code: s.name for s in stocks}

        for p in preds:
            predictions.append({
                "code": p.code,
                "name": stock_map.get(p.code, ""),
                "predicted_return": p.predicted_return,
                "prediction_rank": p.prediction_rank,
                "confidence": p.confidence,
            })

        # Factor means for top/bottom 10
        fs_codes = [p.code for p in preds[:10]] + [p.code for p in preds[-10:]]
        fs_data = (
            db.query(FactorScore)
            .filter(FactorScore.trade_date == latest_date, FactorScore.code.in_(fs_codes))
            .all()
        )
        fs_map = {f.code: f for f in fs_data}

        def factor_means(code_list):
            scores = []
            for c in code_list:
                f = fs_map.get(c)
                if f:
                    scores.append([
                        f.value_score or 0, f.quality_score or 0,
                        f.momentum_score or 0, f.volatility_score or 0,
                        f.composite_score or 0,
                    ])
            if not scores:
                return [0, 0, 0, 0, 0]
            arr = np.array(scores)
            return arr.mean(axis=0).tolist()

        top_factors = factor_means([p.code for p in preds[:10]])
        bot_factors = factor_means([p.code for p in preds[-10:]])

        # Distribution
        vals = [p.predicted_return for p in preds if p.predicted_return is not None]
        if vals:
            hist, bins = np.histogram(vals, bins=20)
            pred_dist["bins"] = [f"{bins[i]:.2f}" for i in range(len(bins) - 1)]
            pred_dist["counts"] = hist.tolist()

    return templates.TemplateResponse("predictions.html", {
        "request": request,
        "page": "predictions",
        "stats": stats,
        "latest_date": latest_date,
        "predictions": predictions,
        "total": total,
        "pred_dist": pred_dist,
        "top_factors": top_factors,
        "bot_factors": bot_factors,
    })


@router.get("/factors")
def factors_page(request: Request, db: Session = Depends(get_db)):
    """Factor analysis page."""
    stats = _get_stats(db)
    latest_date = db.query(func.max(FactorScore.trade_date)).scalar()
    top_stocks = []
    factor_dist = {}
    corr_data = {"labels": [], "values": []}

    if latest_date:
        # Top stocks by composite score
        fs_rows = (
            db.query(FactorScore)
            .filter(FactorScore.trade_date == latest_date)
            .order_by(FactorScore.composite_score.desc())
            .limit(20)
            .all()
        )
        if fs_rows:
            codes = [r.code for r in fs_rows]
            stock_map = {s.code: s.name for s in db.query(Stock).filter(Stock.code.in_(codes)).all()}
            for r in fs_rows:
                top_stocks.append({
                    "code": r.code,
                    "name": stock_map.get(r.code, ""),
                    "value_score": r.value_score,
                    "quality_score": r.quality_score,
                    "momentum_score": r.momentum_score,
                    "volatility_score": r.volatility_score,
                    "composite_score": r.composite_score,
                })

        # Factor distribution (sorted by percentile for line chart)
        all_fs = (
            db.query(FactorScore)
            .filter(FactorScore.trade_date == latest_date)
            .all()
        )
        if all_fs:
            for key in ["value_score", "quality_score", "momentum_score", "volatility_score", "composite_score"]:
                vals = sorted([getattr(r, key) for r in all_fs if getattr(r, key) is not None])
                if len(vals) > 50:
                    idx = np.linspace(0, len(vals) - 1, 100, dtype=int)
                    factor_dist[key.replace("_score", "")] = [vals[i] for i in idx]

        # Correlation matrix
        factor_names = ["value", "quality", "momentum", "volatility", "composite"]
        arr = np.array([
            [getattr(r, f"{n}_score") or np.nan for n in factor_names]
            for r in all_fs
        ])
        mask = ~np.isnan(arr).any(axis=1)
        if mask.sum() > 10:
            corr = np.corrcoef(arr[mask].T)
            corr_data["labels"] = factor_names
            for i in range(len(factor_names)):
                for j in range(len(factor_names)):
                    corr_data["values"].append([i, j, round(float(corr[i][j]), 3)])

    return templates.TemplateResponse("factors.html", {
        "request": request,
        "page": "factors",
        "stats": stats,
        "top_stocks": top_stocks,
        "factor_dist": factor_dist,
        "corr_data": corr_data,
    })


@router.get("/backtest")
def backtest_page(request: Request, db: Session = Depends(get_db)):
    """Backtest results page."""
    backtests = (
        db.query(BacktestResult)
        .order_by(BacktestResult.created_at.desc())
        .all()
    )

    return templates.TemplateResponse("backtest.html", {
        "request": request,
        "page": "backtest",
        "backtests": backtests,
    })


@router.get("/data")
def data_status_page(request: Request, db: Session = Depends(get_db)):
    """Data status and sync page."""
    stats = _get_stats(db)

    # Market breakdown
    markets = db.query(Stock.market, func.count(Stock.code)).group_by(Stock.market).all()
    market_stats = []
    for market, total in markets:
        with_quotes = (
            db.query(func.count(func.distinct(DailyQuote.code)))
            .join(Stock, DailyQuote.code == Stock.code)
            .filter(Stock.market == market)
            .scalar()
        ) or 0
        pct = (with_quotes / total * 100) if total > 0 else 0
        market_stats.append({
            "market": market,
            "total": total,
            "with_quotes": with_quotes,
            "pct": round(pct, 1),
        })

    # Daily quote counts (last 30 days)
    daily_counts = {"dates": [], "counts": []}
    for i in range(30, 0, -1):
        d = (date.today() - timedelta(days=i)).isoformat()
        cnt = db.query(func.count(DailyQuote.id)).filter(DailyQuote.trade_date == d).scalar() or 0
        daily_counts["dates"].append(d[-5:])
        daily_counts["counts"].append(cnt)

    return templates.TemplateResponse("data_status.html", {
        "request": request,
        "page": "data",
        "stats": stats,
        "market_stats": market_stats,
        "daily_counts": daily_counts,
    })
