"""Adversarial AI review — Bull/Bear debate + 5-step decision pipeline.

Inspired by TradingAgents multi-agent LLM architecture:
  - Bull Analyst: presents bullish thesis
  - Bear Analyst: presents bearish counter-thesis
  - Moderator: synthesizes debate into a structured rating

Uses DeepSeek API (OpenAI-compatible). Degrades gracefully without API key.
"""

import json
import logging
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd
import requests
from sqlalchemy.orm import Session

from ..core.config import settings
from ..core.database import SessionLocal

logger = logging.getLogger(__name__)

# ── Prompt templates ─────────────────────────────────────────────────

BULL_ANALYST_PROMPT = """你是一位A股多头分析师。请针对以下股票，从乐观角度分析其投资价值。

股票代码: {code}
股票名称: {name}
所属行业: {industry}

近期行情数据:
- 最新收盘价: {close_price} 元
- 5日涨跌幅: {pct_5d:+.2f}%
- 20日涨跌幅: {pct_20d:+.2f}%
- 60日涨跌幅: {pct_60d:+.2f}%
- 市盈率(PE): {pe}
- 市净率(PB): {pb}
- ROE: {roe}%

技术指标:
- RSI(14): {rsi}
- 量比(5/20): {vol_ratio}
- MACD形态: {macd_signal}
- 价格在20日区间位置: {price_pos}

资金流向:
- 主力净流入: {main_inflow} 万元
- 北向资金净流入: {north_inflow} 万元

请从以下角度分析看多理由:
1. 估值合理性
2. 技术面信号
3. 资金面支撑
4. 行业前景
5. 短期催化剂

请用中文回答，控制在200字内，直接给出核心观点，不要客套话。"""

BEAR_ANALYST_PROMPT = """你是一位A股空头分析师。请针对以下股票，从风险角度分析其潜在问题。

股票代码: {code}
股票名称: {name}
所属行业: {industry}

近期行情数据:
- 最新收盘价: {close_price} 元
- 5日涨跌幅: {pct_5d:+.2f}%
- 20日涨跌幅: {pct_20d:+.2f}%
- 60日涨跌幅: {pct_60d:+.2f}%
- 市盈率(PE): {pe}
- 市净率(PB): {pb}
- ROE: {roe}%

技术指标:
- RSI(14): {rsi}
- 量比(5/20): {vol_ratio}
- MACD形态: {macd_signal}
- 价格在20日区间位置: {price_pos}

资金流向:
- 主力净流入: {main_inflow} 万元
- 北向资金净流入: {north_inflow} 万元

请从以下角度分析看空理由:
1. 估值风险
2. 技术面隐忧
3. 资金面隐患
4. 行业风险
5. 短期利空因素

请用中文回答，控制在200字内，直接给出核心风险点，不要客套话。"""

MODERATOR_PROMPT = """你是一位A股量化投资决策官。下面是一只股票的看多和看空分析，请综合判断给出评级。

股票代码: {code}
股票名称: {name}

=== 看多观点 ===
{bull_view}

=== 看空观点 ===
{bear_view}

请给出以下格式的JSON决策（只输出JSON，不要其他内容）:
{{
    "rating": 3,
    "rating_label": "持有",
    "confidence": 0.65,
    "decision": "hold",
    "position_pct": 10,
    "key_reason": "综合来看，多空因素均衡，建议维持现有仓位观察。",
    "risk_flags": ["估值偏高"],
    "upside_catalysts": ["业绩超预期"]
}}

评级标准:
- 5: 强烈看多 (可重仓, 15-20%仓位)
- 4: 看多 (可加仓, 10-15%仓位)
- 3: 持有/中性 (维持现有, 5-10%仓位)
- 2: 看空 (减仓, 0-5%仓位)
- 1: 强烈看空 (清仓, 0%仓位)

decision: buy / add / hold / reduce / sell
position_pct: 建议仓位占比(%)
risk_flags: 需要关注的风险点
upside_catalysts: 潜在的上涨催化剂"""

MARKET_ENV_PROMPT = """你是一位宏观策略分析师。请根据以下市场环境数据判断当前A股市场状况。

日期: {trade_date}
市场宽度:
- 上涨家数: {advancing}
- 下跌家数: {declining}
- 涨跌比: {adv_decl_ratio}
- 涨停家数: {limit_up}
- 跌停家数: {limit_down}

指数表现:
- 上证指数近5日: {index_5d:+.2f}%
- 上证指数近20日: {index_20d:+.2f}%

请用JSON格式输出市场环境评估（只输出JSON）:
{{
    "market_regime": "bull",
    "regime_label": "强势上涨",
    "risk_level": "low",
    "suggested_exposure": 80,
    "summary": "市场整体强势，量能配合良好，建议积极参与。",
    "key_risks": ["外围市场波动"],
    "sector_preference": ["科技", "新能源"]
}}

market_regime: bull / neutral / bear
risk_level: low / medium / high
suggested_exposure: 建议总仓位占比(%)"""


class AIReviewer:
    """Adversarial AI review for stock trading decisions.

    Uses a 2-stage pipeline:
      Stage 1: Bull/Bear debate → synthesize rating
      Stage 2: Market environment + rating → final decision

    Falls back to rule-based scoring when API key is unavailable.
    """

    def __init__(self, db: Session | None = None):
        self._db = db
        self._api_key = settings.DEEPSEEK_API_KEY
        self._base_url = settings.DEEPSEEK_BASE_URL
        self._model = settings.AI_MODEL
        self._enabled = bool(self._api_key)

    def _get_db(self) -> Session:
        return self._db or SessionLocal()

    # ── Public API ─────────────────────────────────────────────────

    def review_stock(
        self, code: str, trade_date: str, context: dict | None = None
    ) -> dict:
        """Run full adversarial review for a single stock.

        Returns:
            {
                "code": str,
                "trade_date": str,
                "rating": 1-5,
                "rating_label": str,
                "confidence": 0-1,
                "decision": "buy/add/hold/reduce/sell",
                "position_pct": float,
                "bull_view": str,
                "bear_view": str,
                "key_reason": str,
                "risk_flags": [str],
                "upside_catalysts": [str],
                "method": "ai" | "rule",
            }
        """
        if not self._enabled:
            return self._rule_based_review(code, trade_date, context)

        try:
            stock_context = self._gather_stock_context(code, trade_date, context)
            if stock_context is None:
                return self._rule_based_review(code, trade_date, context)

            # Stage 1: Bull/Bear debate
            bull_view = self._call_llm(
                BULL_ANALYST_PROMPT.format(**stock_context),
                max_tokens=400,
            )
            bear_view = self._call_llm(
                BEAR_ANALYST_PROMPT.format(**stock_context),
                max_tokens=400,
            )

            if not bull_view or not bear_view:
                return self._rule_based_review(code, trade_date, context)

            # Stage 2: Moderator synthesis
            moderator_input = MODERATOR_PROMPT.format(
                code=code,
                name=stock_context.get("name", ""),
                bull_view=bull_view,
                bear_view=bear_view,
            )
            moderator_raw = self._call_llm(moderator_input, max_tokens=500)

            decision = self._parse_decision(moderator_raw)
            decision["code"] = code
            decision["trade_date"] = trade_date
            decision["bull_view"] = bull_view
            decision["bear_view"] = bear_view
            decision["method"] = "ai"
            return decision

        except Exception as e:
            logger.warning(f"AI review failed for {code}: {e}")
            return self._rule_based_review(code, trade_date, context)

    def assess_market_environment(self, trade_date: str) -> dict:
        """Assess overall market environment."""
        if not self._enabled:
            return {
                "market_regime": "neutral",
                "risk_level": "medium",
                "suggested_exposure": 50,
                "method": "rule",
            }

        try:
            env_context = self._gather_market_context(trade_date)
            prompt = MARKET_ENV_PROMPT.format(**env_context)
            raw = self._call_llm(prompt, max_tokens=300)

            try:
                # Extract JSON
                json_start = raw.find("{")
                json_end = raw.rfind("}") + 1
                if json_start >= 0 and json_end > json_start:
                    return json.loads(raw[json_start:json_end])
            except (json.JSONDecodeError, ValueError):
                pass
            return {"market_regime": "neutral", "method": "rule"}
        except Exception as e:
            logger.warning(f"Market environment assessment failed: {e}")
            return {"market_regime": "neutral", "method": "rule"}

    def batch_review(
        self, codes: list[str], trade_date: str, top_n: int = 5
    ) -> list[dict]:
        """Review multiple stocks and return top-N rated."""
        results = []
        for code in codes:
            review = self.review_stock(code, trade_date)
            results.append(review)
        results.sort(key=lambda r: (r.get("rating", 0), r.get("confidence", 0)), reverse=True)
        return results[:top_n]

    # ── LLM helpers ──────────────────────────────────────────────

    def _call_llm(self, prompt: str, max_tokens: int = 500) -> str | None:
        """Call DeepSeek API (OpenAI-compatible)."""
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
                        {"role": "system", "content": "你是一位专业的A股量化投资分析师。请用中文回答，保持客观理性。"},
                        {"role": "user", "content": prompt},
                    ],
                    "max_tokens": max_tokens,
                    "temperature": 0.3,
                },
                timeout=60,
            )
            if resp.status_code == 200:
                return resp.json()["choices"][0]["message"]["content"]
            logger.warning(f"DeepSeek API error {resp.status_code}: {resp.text[:200]}")
            return None
        except requests.Timeout:
            logger.warning("DeepSeek API timeout")
            return None
        except Exception as e:
            logger.warning(f"DeepSeek API call failed: {e}")
            return None

    @staticmethod
    def _parse_decision(raw: str | None) -> dict:
        """Parse moderator JSON output, with fallback."""
        default = {
            "rating": 3, "rating_label": "持有", "confidence": 0.5,
            "decision": "hold", "position_pct": 5,
            "key_reason": "AI分析暂不可用",
            "risk_flags": [], "upside_catalysts": [],
        }
        if not raw:
            return default

        try:
            # Extract JSON block
            json_start = raw.find("{")
            json_end = raw.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                parsed = json.loads(raw[json_start:json_end])
                for k in default:
                    if k not in parsed:
                        parsed[k] = default[k]
                return parsed
        except (json.JSONDecodeError, ValueError):
            pass
        return default

    # ── Context gathering ────────────────────────────────────────

    def _gather_stock_context(
        self, code: str, trade_date: str, extra: dict | None = None
    ) -> dict | None:
        """Gather all data needed for stock-level AI review."""
        db = self._get_db()
        from ..models.market import DailyQuote
        from ..models.stock import Stock
        from ..models.finance import FinancialIndicator, FactorScore, FundFlow

        stock = db.get(Stock, code)
        if not stock:
            return None

        # Latest quote
        quote = (
            db.query(DailyQuote)
            .filter(DailyQuote.code == code, DailyQuote.trade_date == trade_date)
            .first()
        )

        # Multi-day returns
        quotes = (
            db.query(DailyQuote.trade_date, DailyQuote.close, DailyQuote.turnover)
            .filter(DailyQuote.code == code)
            .order_by(DailyQuote.trade_date.desc())
            .limit(120)
            .all()
        )
        closes = [row.close for row in quotes if row.close and row.close > 0]

        def _pct(days):
            if len(closes) > days and closes[days]:
                return (closes[0] - closes[days]) / closes[days] * 100
            return 0.0

        # Financials
        fi = (
            db.query(FinancialIndicator)
            .filter(FinancialIndicator.code == code)
            .order_by(FinancialIndicator.report_date.desc())
            .first()
        )

        # Factor scores
        fs = (
            db.query(FactorScore)
            .filter(FactorScore.code == code, FactorScore.trade_date == trade_date)
            .first()
        )

        # Fund flow
        ff = (
            db.query(FundFlow)
            .filter(FundFlow.code == code, FundFlow.trade_date == trade_date)
            .first()
        )

        rsi_val = f"{fs.momentum_score:.1f}" if (fs and fs.momentum_score is not None) else "N/A"
        vol_ratio = f"{fs.volatility_score:.2f}" if (fs and fs.volatility_score is not None) else "N/A"
        price_pos = f"{fs.composite_score:.2f}" if (fs and fs.composite_score is not None) else "N/A"

        # MACD signal (simplified)
        macd_signal = "中性"
        if len(closes) >= 26:
            pct_12d = (closes[0] - closes[min(11, len(closes) - 1)]) / closes[min(11, len(closes) - 1)] * 100
            pct_26d = (closes[0] - closes[min(25, len(closes) - 1)]) / closes[min(25, len(closes) - 1)] * 100
            if pct_12d > pct_26d:
                macd_signal = "金叉偏多" if pct_12d > 0 else "低位金叉"
            else:
                macd_signal = "死叉偏空" if pct_12d < 0 else "高位死叉"

        return {
            "code": code,
            "name": stock.name or code,
            "industry": stock.industry or "未知",
            "close_price": f"{closes[0]:.2f}" if closes else "N/A",
            "pct_5d": _pct(5),
            "pct_20d": _pct(20),
            "pct_60d": _pct(60),
            "pe": f"{fi.pe:.1f}" if fi and fi.pe else "N/A",
            "pb": f"{fi.pb:.1f}" if fi and fi.pb else "N/A",
            "roe": f"{fi.roe:.1f}" if fi and fi.roe else "N/A",
            "rsi": rsi_val,
            "vol_ratio": vol_ratio,
            "macd_signal": macd_signal,
            "price_pos": price_pos,
            "main_inflow": f"{ff.main_net_inflow:.0f}" if ff and ff.main_net_inflow else "N/A",
            "north_inflow": f"{ff.north_bound_net:.0f}" if ff and ff.north_bound_net else "N/A",
        }

    def _gather_market_context(self, trade_date: str) -> dict:
        """Gather market-wide data for environment assessment."""
        from ..models.market import DailyQuote
        from sqlalchemy import func

        db = self._get_db()

        # Count advancing/declining
        quotes = (
            db.query(DailyQuote.pct_change)
            .filter(DailyQuote.trade_date == trade_date)
            .all()
        )
        changes = [q.pct_change for q in quotes if q.pct_change is not None]
        advancing = sum(1 for c in changes if c > 0)
        declining = sum(1 for c in changes if c < 0)
        limit_up = sum(1 for c in changes if c >= 9.9)
        limit_down = sum(1 for c in changes if c <= -9.9)

        # Index performance via average daily return over lookback windows
        index_5d = 0.0
        index_20d = 0.0
        for lookback, target in [(5, "index_5d"), (20, "index_20d")]:
            start_d = (date.fromisoformat(trade_date) - timedelta(days=lookback + 5)).isoformat()
            row = (
                db.query(func.avg(DailyQuote.pct_change))
                .filter(DailyQuote.trade_date >= start_d, DailyQuote.trade_date <= trade_date)
                .first()
            )
            if row and row[0] is not None:
                if target == "index_5d":
                    index_5d = round(float(row[0]) * lookback, 2)
                else:
                    index_20d = round(float(row[0]) * lookback, 2)

        return {
            "trade_date": trade_date,
            "advancing": advancing,
            "declining": declining,
            "adv_decl_ratio": round(advancing / max(declining, 1), 2),
            "limit_up": limit_up,
            "limit_down": limit_down,
            "index_5d": index_5d,
            "index_20d": index_20d,
        }

    # ── Rule-based fallback ─────────────────────────────────────

    def _rule_based_review(
        self, code: str, trade_date: str, context: dict | None = None
    ) -> dict:
        """Rule-based scoring as fallback when AI is unavailable."""
        db = self._get_db()
        from ..models.finance import FactorScore

        fs = (
            db.query(FactorScore)
            .filter(FactorScore.code == code, FactorScore.trade_date == trade_date)
            .first()
        )

        score = 3
        risk_flags = []
        catalysts = []

        if fs:
            if fs.composite_score and fs.composite_score > 1.0:
                score = min(5, score + 1)
                catalysts.append("综合因子得分较高")
            elif fs.composite_score and fs.composite_score < -1.0:
                score = max(1, score - 1)
                risk_flags.append("综合因子得分偏低")

            if fs.momentum_score and fs.momentum_score > 1.5:
                score = min(5, score + 1)
                catalysts.append("动量强劲")
            if fs.momentum_score and fs.momentum_score < -1.5:
                score = max(1, score - 1)
                risk_flags.append("动能衰竭")

            if fs.value_score and fs.value_score > 1.0:
                score = min(5, score + 1)
                catalysts.append("估值优势明显")
            if fs.quality_score and fs.quality_score < -1.0:
                risk_flags.append("基本面偏弱")

        rating_labels = {1: "强烈看空", 2: "看空", 3: "持有", 4: "看多", 5: "强烈看多"}
        decisions = {1: "sell", 2: "reduce", 3: "hold", 4: "add", 5: "buy"}
        position_pcts = {1: 0, 2: 3, 3: 5, 4: 10, 5: 15}

        return {
            "code": code,
            "trade_date": trade_date,
            "rating": score,
            "rating_label": rating_labels[score],
            "confidence": 0.4,
            "decision": decisions[score],
            "position_pct": position_pcts[score],
            "bull_view": "",
            "bear_view": "",
            "key_reason": f"基于规则评分: composite={fs.composite_score:.2f}" if fs else "数据不足",
            "risk_flags": risk_flags,
            "upside_catalysts": catalysts,
            "method": "rule",
        }
