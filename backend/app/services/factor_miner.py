"""LLM-driven automatic factor mining.

Inspired by RD-Agent: hypothesis → code → backtest → feedback loop.

Flow:
  1. LLM proposes factor hypotheses based on market concepts
  2. Auto-generate Python code for the factor
  3. Sandbox-execute the code to compute factor values
  4. Backtest: compute IC, rank IC, correlation with existing factors
  5. If IC > threshold → add to factor library
  6. If IC < threshold → send feedback to LLM for refinement
  7. Dedup: drop factors with >0.8 correlation to existing factors
"""

import json
import logging
import sys
from datetime import date, timedelta
from io import StringIO
from typing import Any

import numpy as np
import pandas as pd
import requests
from sqlalchemy.orm import Session

from ..core.config import settings
from ..core.database import SessionLocal

logger = logging.getLogger(__name__)

# ── Prompt templates ─────────────────────────────────────────────────

FACTOR_HYPOTHESIS_PROMPT = """你是一位量化因子研究员。请提出一个新的A股选股因子。

要求:
1. 因子要有金融学逻辑支撑（不是纯数据挖掘）
2. 因子值越高，预期未来收益越高（正向因子）
3. 尽量不与常见因子重复（动量、换手率、RSI、波动率等）
4. 可以使用OHLCV数据（开高低收量额），也可以结合财务数据

请用以下JSON格式回答（只输出JSON）:
{{
    "factor_name": "volume_price_trend",
    "chinese_name": "量价趋势因子",
    "description": "衡量成交量与价格趋势的一致性。当量价齐升时，趋势确认度高；量价背离时，趋势可能反转。",
    "formula": "corr(volume, close, 20) * sign(mom_20d)",
    "python_code": "def compute(df):\\n    '''df has columns: open, high, low, close, volume, amount, turnover, pct_change\\n    Returns: pd.Series with same index as df'''\\n    vol_price_corr = df['volume'].rolling(20).corr(df['close'])\\n    mom_20d = df['close'].pct_change(20)\\n    return vol_price_corr * np.sign(mom_20d)",
    "expected_ic": 0.03,
    "universe": "all"
}}"""

FACTOR_REFINEMENT_PROMPT = """你之前提出的因子回测结果不理想。请分析原因并改进。

原始因子:
{original_factor}

回测结果:
- IC均值: {ic_mean:.4f}
- IC标准差: {ic_std:.4f}
- IC_IR: {ic_ir:.4f}
- 胜率(IC>0): {ic_win_rate:.1%}
- 与现有因子最大相关性: {max_corr:.2f}

请分析失败原因并提出改进方案，用JSON格式回答:
{{
    "failure_analysis": "IC过低的原因分析...",
    "improved_factor_name": "new_name",
    "improved_code": "def compute(df):\\n    ...",
    "expected_improvement": "改进后的预期效果"
}}"""


class FactorMiner:
    """LLM-driven automatic factor mining pipeline."""

    def __init__(self, db: Session | None = None):
        self._db = db
        self._api_key = settings.DEEPSEEK_API_KEY
        self._base_url = settings.DEEPSEEK_BASE_URL
        self._model = settings.AI_MODEL
        self._enabled = bool(self._api_key)
        self._mined_factors: list[dict] = []

    def _get_db(self) -> Session:
        return self._db or SessionLocal()

    # ── Main pipeline ─────────────────────────────────────────────

    def mine(
        self,
        trade_date: str,
        n_attempts: int = 10,
        min_ic: float = 0.02,
        max_corr: float = 0.7,
    ) -> list[dict]:
        """Run the full mining pipeline.

        Args:
            trade_date: Base date for factor computation.
            n_attempts: Max number of factor hypotheses to try.
            min_ic: Minimum absolute IC to keep a factor.
            max_corr: Maximum correlation with existing factors (dedup).

        Returns:
            List of accepted factors with their metrics.
        """
        db = self._get_db()

        # Get universe and price data for backtesting
        codes = self._get_universe_codes(db)

        # Load existing factor values for dedup
        existing_factors = self._load_existing_factors(db, codes, trade_date)

        accepted = []
        attempts = 0

        while attempts < n_attempts:
            attempts += 1
            logger.info(f"Mining attempt {attempts}/{n_attempts}")

            # Step 1: Generate hypothesis
            if attempts == 1:
                hypothesis = self._generate_hypothesis()
            else:
                # Refine the worst-performing accepted or random one
                hypothesis = self._refine_hypothesis(
                    accepted[-1] if accepted else self._mined_factors[-1] if self._mined_factors else {}
                )

            if not hypothesis:
                continue

            # Step 2: Execute factor code
            factor_values = self._execute_factor(hypothesis, codes, trade_date, db)
            if factor_values is None or len(factor_values) < 50:
                logger.warning(f"Factor {hypothesis.get('factor_name')} execution failed")
                continue

            # Step 3: Compute IC
            ic_metrics = self._compute_ic(factor_values, codes, trade_date, db)

            # Step 4: Check correlation with existing factors
            max_existing_corr = self._max_correlation(factor_values, existing_factors)
            ic_metrics["max_existing_corr"] = max_existing_corr

            result = {
                **hypothesis,
                "ic_mean": ic_metrics["ic_mean"],
                "ic_std": ic_metrics["ic_std"],
                "ic_ir": ic_metrics["ic_ir"],
                "ic_win_rate": ic_metrics["ic_win_rate"],
                "max_existing_corr": max_existing_corr,
                "accepted": False,
                "attempts": attempts,
            }

            self._mined_factors.append(result)

            # Step 5: Accept or reject
            abs_ic = abs(ic_metrics["ic_mean"])
            if abs_ic >= min_ic and max_existing_corr <= max_corr:
                result["accepted"] = True
                accepted.append(result)
                # Add to existing factors for future dedup
                clean_values = factor_values.dropna()
                existing_factors[hypothesis.get("factor_name", f"mined_{len(accepted)}")] = clean_values
                logger.info(
                    f"ACCEPTED: {hypothesis.get('factor_name')} "
                    f"IC={ic_metrics['ic_mean']:.4f}, corr={max_existing_corr:.2f}"
                )
            else:
                reason = f"IC={abs_ic:.4f}<{min_ic}" if abs_ic < min_ic else f"corr={max_existing_corr:.2f}>{max_corr}"
                logger.info(f"REJECTED: {hypothesis.get('factor_name')} ({reason})")

                # Refine rejected factors
                if attempts < n_attempts - 1:
                    refined = self._refine_hypothesis(result)
                    if refined:
                        hypothesis = refined  # use refined in next iteration

        return accepted

    # ── Step 1: Hypothesis generation ────────────────────────────

    def _generate_hypothesis(self) -> dict | None:
        """Use LLM to generate a new factor hypothesis."""
        if not self._enabled:
            return self._rule_based_hypothesis()

        raw = self._call_llm(FACTOR_HYPOTHESIS_PROMPT, max_tokens=600)
        return self._parse_json(raw)

    def _refine_hypothesis(self, failed: dict) -> dict | None:
        """Ask LLM to improve a failed factor."""
        if not self._enabled or not failed:
            return self._generate_hypothesis()

        prompt = FACTOR_REFINEMENT_PROMPT.format(
            original_factor=json.dumps(failed, ensure_ascii=False, indent=2),
            ic_mean=failed.get("ic_mean", 0),
            ic_std=failed.get("ic_std", 0),
            ic_ir=failed.get("ic_ir", 0),
            ic_win_rate=failed.get("ic_win_rate", 0),
            max_corr=failed.get("max_existing_corr", 0),
        )
        raw = self._call_llm(prompt, max_tokens=600)
        parsed = self._parse_json(raw)
        if parsed and "improved_code" in parsed:
            parsed["python_code"] = parsed.pop("improved_code")
            parsed["factor_name"] = parsed.get("improved_factor_name", parsed.get("factor_name", "refined"))
        return parsed or self._rule_based_hypothesis()

    def _rule_based_hypothesis(self) -> dict:
        """Generate simple rule-based factor when LLM is unavailable."""
        templates = [
            {
                "factor_name": "hl_amplitude",
                "chinese_name": "高低价振幅因子",
                "description": "(high-low)/close 的20日均值，反映日内波动程度",
                "python_code": (
                    "def compute(df):\n"
                    "    amp = (df['high'] - df['low']) / df['close']\n"
                    "    return -amp.rolling(20).mean()  # lower amplitude = more stable"
                ),
            },
            {
                "factor_name": "close_position_5d",
                "chinese_name": "5日收盘位置因子",
                "description": "收盘价在5日高低区间的相对位置",
                "python_code": (
                    "def compute(df):\n"
                    "    h5 = df['high'].rolling(5).max()\n"
                    "    l5 = df['low'].rolling(5).min()\n"
                    "    return (df['close'] - l5) / (h5 - l5 + 1e-9)"
                ),
            },
            {
                "factor_name": "volume_reversal",
                "chinese_name": "缩量反弹因子",
                "description": "下跌后缩量企稳，可能是反转信号",
                "python_code": (
                    "def compute(df):\n"
                    "    ret_5d = df['close'].pct_change(5)\n"
                    "    vol_ratio = df['volume'] / df['volume'].rolling(20).mean()\n"
                    "    return -ret_5d * (1 / (vol_ratio + 0.1))  # negative ret + low vol = reversal"
                ),
            },
        ]
        import random
        return random.choice(templates)

    # ── Step 2: Factor execution ─────────────────────────────────

    def _execute_factor(
        self, hypothesis: dict, codes: list[str], trade_date: str, db: Session
    ) -> pd.Series | None:
        """Execute the factor's Python code in a sandbox and return computed values."""
        code_str = hypothesis.get("python_code", "")
        if not code_str:
            return None

        # Extract the compute function
        try:
            local_ns = {}
            exec(code_str, {"np": np, "pd": pd}, local_ns)
            compute_fn = local_ns.get("compute")
            if not callable(compute_fn):
                return None
        except Exception as e:
            logger.warning(f"Factor code compilation failed: {e}")
            return None

        # Load price data
        from ..models.market import DailyQuote
        from .factor_engine import FactorEngine

        start = (date.fromisoformat(trade_date) - timedelta(days=365)).isoformat()
        engine = FactorEngine(db)
        price_data = engine._load_price_data(db, codes, start, trade_date)

        results = {}
        for code, df in price_data.items():
            if df is None or len(df) < 20:
                continue
            try:
                factor_series = compute_fn(df)
                if isinstance(factor_series, pd.Series) and trade_date in factor_series.index:
                    val = factor_series.loc[trade_date]
                    if not pd.isna(val):
                        results[code] = float(val)
            except Exception:
                continue

        if len(results) < 50:
            return None

        return pd.Series(results)

    # ── Step 3: IC computation ───────────────────────────────────

    def _compute_ic(
        self, factor_values: pd.Series, codes: list[str], trade_date: str, db: Session
    ) -> dict:
        """Compute daily IC (Information Coefficient) time series for a factor."""
        from ..models.market import DailyQuote

        trading_dates = sorted(
            r[0] for r in db.query(DailyQuote.trade_date)
            .filter(DailyQuote.trade_date <= trade_date)
            .distinct()
            .order_by(DailyQuote.trade_date.desc())
            .limit(252)
            .all()
        )

        ic_list = []
        for td in trading_dates[1:-5]:  # avoid edges
            factor_at_td = {}
            forward_rets = {}

            for code in factor_values.index:
                quotes = (
                    db.query(DailyQuote.close, DailyQuote.trade_date)
                    .filter(DailyQuote.code == code)
                    .order_by(DailyQuote.trade_date)
                    .all()
                )

                # Find quote at td
                td_quote = None
                for q in quotes:
                    if q.trade_date == td:
                        td_quote = q
                        break
                if not td_quote or not td_quote.close:
                    continue

                # Find quote 5 days later
                td_idx = next((i for i, q in enumerate(quotes) if q.trade_date == td), None)
                if td_idx is None or td_idx + 5 >= len(quotes):
                    continue

                fut = quotes[td_idx + 5]
                if fut.close and fut.close > 0 and td_quote.close > 0:
                    factor_at_td[code] = float(factor_values.get(code, np.nan))
                    forward_rets[code] = (fut.close - td_quote.close) / td_quote.close

            if len(factor_at_td) < 30:
                continue

            try:
                from scipy.stats import spearmanr
                f_vals = pd.Series(factor_at_td).dropna()
                r_vals = pd.Series(forward_rets).dropna()
                common = f_vals.index.intersection(r_vals.index)
                if len(common) >= 30:
                    ic, _ = spearmanr(f_vals[common], r_vals[common])
                    if not np.isnan(ic):
                        ic_list.append(ic)
            except Exception:
                continue

        if not ic_list:
            return {"ic_mean": 0.0, "ic_std": 0.0, "ic_ir": 0.0, "ic_win_rate": 0.0}

        ic_arr = np.array(ic_list)
        return {
            "ic_mean": round(float(ic_arr.mean()), 4),
            "ic_std": round(float(ic_arr.std()), 4),
            "ic_ir": round(float(ic_arr.mean() / ic_arr.std()) if ic_arr.std() > 0 else 0, 4),
            "ic_win_rate": round(float((ic_arr > 0).mean()), 4),
        }

    # ── Step 4: Correlation dedup ────────────────────────────────

    def _max_correlation(
        self, new_factor: pd.Series, existing: dict[str, pd.Series]
    ) -> float:
        """Compute max absolute correlation with existing factors."""
        if not existing:
            return 0.0

        new_clean = new_factor.dropna()
        max_corr = 0.0

        for name, series in existing.items():
            common = new_clean.index.intersection(series.dropna().index)
            if len(common) < 30:
                continue
            try:
                corr = abs(new_clean[common].corr(series[common]))
                if not np.isnan(corr) and corr > max_corr:
                    max_corr = corr
            except Exception:
                continue

        return round(max_corr, 4)

    # ── Helpers ──────────────────────────────────────────────────

    def _get_universe_codes(self, db: Session) -> list[str]:
        from ..models.stock import Stock

        return [
            r[0] for r in db.query(Stock.code)
            .filter(Stock.is_active == True)
            .order_by(Stock.code).all()
        ]

    def _load_existing_factors(
        self, db: Session, codes: list[str], trade_date: str
    ) -> dict[str, pd.Series]:
        """Load existing factor scores for dedup."""
        from ..models.finance import FactorScore

        rows = (
            db.query(FactorScore)
            .filter(FactorScore.trade_date == trade_date, FactorScore.code.in_(codes))
            .all()
        )

        result = {}
        if not rows:
            return result

        fields = ["value_score", "quality_score", "momentum_score", "volatility_score", "composite_score"]
        for field in fields:
            series_data = {}
            for r in rows:
                val = getattr(r, field, None)
                if val is not None:
                    series_data[r.code] = val
            if series_data:
                result[field] = pd.Series(series_data)

        return result

    def _call_llm(self, prompt: str, max_tokens: int = 600) -> str | None:
        """Call DeepSeek API."""
        try:
            resp = requests.post(
                f"{self._base_url}/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self._model,
                    "messages": [
                        {"role": "system", "content": "你是一位专业的量化因子研究员。请用中文回答，只输出要求的JSON格式。"},
                        {"role": "user", "content": prompt},
                    ],
                    "max_tokens": max_tokens,
                    "temperature": 0.5,
                },
                timeout=90,
            )
            if resp.status_code == 200:
                return resp.json()["choices"][0]["message"]["content"]
            logger.warning(f"API error: {resp.status_code}")
            return None
        except Exception as e:
            logger.warning(f"LLM call failed: {e}")
            return None

    @staticmethod
    def _parse_json(raw: str | None) -> dict | None:
        if not raw:
            return None
        try:
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(raw[start:end])
        except json.JSONDecodeError:
            pass
        return None
