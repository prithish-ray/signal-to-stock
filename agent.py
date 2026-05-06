"""
Macro-to-Company Transmission Agent
Uses Tavily for search context and Groq for causal reasoning.
Yields structured step events so the Flask SSE endpoint can stream them to the UI.
"""

import os
import json
import re
from groq import Groq
from tavily import TavilyClient

# ---------------------------------------------------------------------------
# Config — pulled from env; hardcoded fallbacks for local dev
# ---------------------------------------------------------------------------
GROQ_API_KEY   = os.environ.get("GROQ_API_KEY")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")
GROQ_MODEL     = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clean_json(raw: str) -> str:
    """Strip markdown fences and leading/trailing whitespace."""
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?", "", raw)
    raw = re.sub(r"```$", "", raw)
    return raw.strip()


def _safe_json(raw: str, fallback: dict) -> dict:
    try:
        return json.loads(_clean_json(raw))
    except Exception:
        # Try to extract the first {...} block
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
        return fallback


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a senior macroeconomic analyst at a top-tier hedge fund.
Your speciality is tracing transmission mechanisms — how macro shocks propagate
through industries and eventually appear on specific company income statements and
balance sheets.

Rules:
- Always explain the causal chain step-by-step (A → B → C).
- Be precise about which financial line item is affected and why.
- Identify the typical reporting lag before the impact is visible in earnings.
- Call out what confirming evidence investors should monitor.
- Return ONLY valid JSON when asked for structured output — no markdown fences."""


class MacroTransmissionAgent:

    def __init__(self):
        self.groq   = Groq(api_key=GROQ_API_KEY)
        self.tavily = TavilyClient(api_key=TAVILY_API_KEY)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _search(self, query: str, max_results: int = 5) -> list[dict]:
        try:
            res = self.tavily.search(
                query=query,
                max_results=max_results,
                search_depth="advanced",
            )
            return res.get("results", [])
        except Exception as e:
            return [{"title": f"Search error: {e}", "content": "", "url": ""}]

    def _complete(self, user_prompt: str, json_mode: bool = False) -> str:
        kwargs = dict(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=4096,
        )
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        resp = self.groq.chat.completions.create(**kwargs)
        return resp.choices[0].message.content

    @staticmethod
    def _fmt_results(results: list[dict]) -> str:
        lines = []
        for r in results:
            title   = r.get("title", "")
            content = r.get("content", "")[:350]
            lines.append(f"[{title}]\n{content}")
        return "\n\n".join(lines)

    # ------------------------------------------------------------------
    # Main analysis generator — yields dicts consumed by Flask SSE
    # ------------------------------------------------------------------

    def analyze(self, macro_event: str):

        # ================================================================
        # STEP 1 — Gather event context via Tavily
        # ================================================================
        yield {
            "type": "step_start",
            "step": 1,
            "title": "Searching for macro event context",
            "detail": f'Running two Tavily queries for "{macro_event}"',
        }

        q1 = macro_event
        q2 = f"{macro_event} economic impact sectors industries"
        r1 = self._search(q1, max_results=5)
        r2 = self._search(q2, max_results=5)

        yield {
            "type": "searches",
            "step": 1,
            "queries": [
                {"query": q1, "results": [r.get("title", "") for r in r1]},
                {"query": q2, "results": [r.get("title", "") for r in r2]},
            ],
        }

        combined_context = self._fmt_results((r1 + r2)[:8])

        # ================================================================
        # STEP 2 — Identify primary industries (first-order)
        # ================================================================
        yield {
            "type": "step_start",
            "step": 2,
            "title": "Identifying primary (first-order) affected industries",
            "detail": "Reasoning about direct causal mechanisms and financial line-item impacts…",
        }

        primary_prompt = f"""Macro event: {macro_event}

Recent news / research context:
{combined_context[:3500]}

Task: Identify 4–6 PRIMARY industries DIRECTLY hit by this macro event.

For EACH industry provide:
1. The exact causal mechanism (how does the event hit this industry — be specific).
2. Which financial line item is affected (revenue, COGS, gross margin, capex, FCF, …).
3. Impact direction + magnitude estimate.
4. Typical reporting lag (in quarters) before the hit appears in earnings.
5. Your confidence level.

Return ONLY this JSON schema (no extra keys, no markdown):
{{
  "event_summary": "<one-sentence interpretation of what this macro event means mechanically>",
  "primary_industries": [
    {{
      "industry": "<name>",
      "mechanism": "<precise causal chain: event → effect on this industry>",
      "line_item_affected": "<specific P&L / balance-sheet line>",
      "impact_direction": "negative | positive | mixed",
      "impact_magnitude": "high | medium | low",
      "lag_quarters": "0–1 | 1–2 | 2–4 | 4+",
      "confidence": "high | medium | low"
    }}
  ]
}}"""

        primary_raw  = self._complete(primary_prompt, json_mode=True)
        primary_data = _safe_json(primary_raw, {"event_summary": "", "primary_industries": []})

        yield {"type": "step_result", "step": 2, "data": primary_data}

        # ================================================================
        # STEP 3 — Second-order / ripple effects
        # ================================================================
        yield {
            "type": "step_start",
            "step": 3,
            "title": "Mapping second-order ripple effects",
            "detail": "Tracing supply-chain, customer, substitution and financial-contagion linkages…",
        }

        p_names = [i["industry"] for i in primary_data.get("primary_industries", [])]
        q3 = f"{macro_event} supply chain spillover effects {' '.join(p_names[:3])}"
        r3 = self._search(q3, max_results=4)

        yield {
            "type": "searches",
            "step": 3,
            "queries": [{"query": q3, "results": [r.get("title", "") for r in r3]}],
        }

        sc_context = self._fmt_results(r3)

        second_prompt = f"""Macro event: {macro_event}

Primary industries and their direct impacts:
{json.dumps(primary_data.get("primary_industries", []), indent=2)}

Additional supply-chain / sector context:
{sc_context[:2500]}

Task: Identify 4–6 SECOND-ORDER industries — those affected *because* the primary industries
are affected. Think about: upstream suppliers, downstream customers, substitution effects,
correlated credit/financial exposures, labour-market spillovers.

For EACH second-order industry provide:
1. Which primary industry triggers it and how (full causal chain: event → primary → secondary).
2. The specific financial line item affected.
3. Impact direction + magnitude.
4. Reporting lag (typically longer than primary).
5. Confidence.

Return ONLY this JSON schema:
{{
  "second_order_industries": [
    {{
      "industry": "<name>",
      "triggered_by": "<primary industry name>",
      "causal_chain": "<step-by-step: event causes X in A, which causes Y in B because …>",
      "line_item_affected": "<specific line>",
      "impact_direction": "negative | positive | mixed",
      "impact_magnitude": "high | medium | low",
      "lag_quarters": "1–2 | 2–4 | 4+",
      "confidence": "high | medium | low"
    }}
  ]
}}"""

        second_raw  = self._complete(second_prompt, json_mode=True)
        second_data = _safe_json(second_raw, {"second_order_industries": []})

        yield {"type": "step_result", "step": 3, "data": second_data}

        # ================================================================
        # STEP 4 — Identify most-exposed listed companies
        # ================================================================
        yield {
            "type": "step_start",
            "step": 4,
            "title": "Identifying most-exposed listed companies",
            "detail": "Searching for specific stocks with highest exposure per sector…",
        }

        all_industries = (
            [(i["industry"], "primary",   i.get("impact_direction", ""))
             for i in primary_data.get("primary_industries", [])]
            + [(i["industry"], "secondary", i.get("impact_direction", ""))
               for i in second_data.get("second_order_industries", [])]
        )

        company_context_parts = []
        search_log = []
        for industry_name, order, direction in all_industries[:7]:
            q = f"publicly listed companies most exposed to {industry_name} {macro_event} stocks earnings"
            results = self._search(q, max_results=3)
            search_log.append({"query": q, "results": [r.get("title", "") for r in results]})
            snippet = self._fmt_results(results)
            company_context_parts.append(
                f"### {industry_name} ({order})\n{snippet}"
            )

        yield {"type": "searches", "step": 4, "queries": search_log}

        company_context = "\n\n".join(company_context_parts)

        company_prompt = f"""Macro event: {macro_event}

Industry impact summary:
Primary:   {json.dumps(primary_data.get("primary_industries", []), indent=2)}
Secondary: {json.dumps(second_data.get("second_order_industries", []), indent=2)}

Company research context:
{company_context[:3500]}

Task: For EACH industry (primary and secondary) identify 2–3 LISTED COMPANIES with the
highest financial exposure to this macro event.

For each company explain:
1. WHY this specific company is especially exposed (revenue mix, geography, cost structure).
2. Which precise financial line item gets hit and by how much (% estimate if possible).
3. What confirming evidence investors should watch — name specific metrics, data releases,
   or management-guidance language to listen for on earnings calls.
4. Ticker symbol (use standard exchange ticker where known).

Return ONLY this JSON schema:
{{
  "company_exposures": [
    {{
      "industry": "<industry name>",
      "order": "primary | secondary",
      "companies": [
        {{
          "name": "<Company Name>",
          "ticker": "<TICK>",
          "exchange": "<NYSE | NASDAQ | LSE | …>",
          "why_exposed": "<specific business-model reason>",
          "financial_impact": "<line item + estimated magnitude>",
          "confirming_evidence": "<what to watch: metric, release, call language>",
          "exposure_level": "high | medium"
        }}
      ]
    }}
  ]
}}"""

        companies_raw  = self._complete(company_prompt, json_mode=True)
        companies_data = _safe_json(companies_raw, {"company_exposures": []})

        yield {"type": "step_result", "step": 4, "data": companies_data}

        # ================================================================
        # STEP 5 — Analyst synthesis
        # ================================================================
        yield {
            "type": "step_start",
            "step": 5,
            "title": "Synthesising analyst narrative",
            "detail": "Writing the final transmission map summary…",
        }

        synthesis_prompt = f"""You have completed a full macro transmission analysis for:
"{macro_event}"

Event summary: {primary_data.get("event_summary", "")}
Primary industries: {json.dumps(primary_data.get("primary_industries", []), indent=2)}
Secondary industries: {json.dumps(second_data.get("second_order_industries", []), indent=2)}
Company exposures: {json.dumps(companies_data.get("company_exposures", []), indent=2)}

Write a concise, analyst-quality synthesis in 3–4 tight paragraphs:

Paragraph 1 — What this macro event REALLY means mechanically (not just a restatement).
Paragraph 2 — The single most important transmission pathway: the key causal chain from
  the macro trigger to an earnings impact. Be specific about mechanism and lag.
Paragraph 3 — The 2–3 companies where the impact will be MOST VISIBLE in reported financials,
  and exactly WHY (name the line item and the expected magnitude).
Paragraph 4 — What confirming or disconfirming evidence investors should track over the
  next 1–4 quarters: specific data releases, company metrics, management language.

Write as a senior analyst. Be direct. No hedging. No bullet points. No markdown."""

        synthesis = self._complete(synthesis_prompt, json_mode=False)

        # ================================================================
        # FINAL — Emit the complete transmission map
        # ================================================================
        yield {
            "type": "transmission_map",
            "event": macro_event,
            "event_summary":          primary_data.get("event_summary", ""),
            "synthesis":              synthesis,
            "primary_industries":     primary_data.get("primary_industries", []),
            "second_order_industries": second_data.get("second_order_industries", []),
            "company_exposures":      companies_data.get("company_exposures", []),
        }
