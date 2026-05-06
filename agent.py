"""
Macro-to-Company Transmission Agent
Uses Tavily for search context and Groq for causal reasoning.
Yields structured step events so the Flask SSE endpoint can stream them to the UI.
British English spelling used throughout user-facing strings.
"""

import os
import json
import re
from groq import Groq
from tavily import TavilyClient
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY   = os.environ.get("GROQ_API_KEY")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")
GROQ_MODEL     = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")


def _clean_json(raw):
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?", "", raw)
    raw = re.sub(r"```$", "", raw)
    return raw.strip()


def _safe_json(raw, fallback):
    try:
        return json.loads(_clean_json(raw))
    except Exception:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
        return fallback


SYSTEM_PROMPT = """You are a senior macroeconomic analyst at a top-tier hedge fund.
Your speciality is tracing transmission mechanisms — how macro events, industry shifts,
supply chain dynamics, and pricing-power changes propagate through industries and
eventually appear on specific company income statements and balance sheets.

Rules:
- Always explain the causal chain step-by-step (A to B to C).
- Be precise about which financial line item is affected and why.
- Identify the typical reporting lag before the impact is visible in earnings.
- Call out what confirming evidence investors should monitor.
- Use British English spelling throughout (e.g. analyse not analyze, behaviour not behavior).
- Return ONLY valid JSON when asked for structured output — no markdown fences."""


class MacroTransmissionAgent:

    def __init__(self):
        self.groq   = Groq(api_key=GROQ_API_KEY)
        self.tavily = TavilyClient(api_key=TAVILY_API_KEY)

    def _search(self, query, max_results=5):
        try:
            res = self.tavily.search(
                query=query,
                max_results=max_results,
                search_depth="advanced",
            )
            return res.get("results", [])
        except Exception as e:
            return [{"title": "Search error: " + str(e), "content": "", "url": ""}]

    def _complete(self, user_prompt, json_mode=False):
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
    def _fmt_results(results):
        lines = []
        for r in results:
            title   = r.get("title", "")
            content = r.get("content", "")[:350]
            lines.append("[" + title + "]\n" + content)
        return "\n\n".join(lines)

    def analyse(self, macro_event):

        # STEP 1 — Gather event context via Tavily
        yield {
            "type": "step_start",
            "step": 1,
            "title": "Searching for signal context",
            "detail": "Running two Tavily queries for \"" + macro_event + "\"",
        }

        q1 = macro_event
        q2 = macro_event + " economic impact sectors industries"
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

        # STEP 2 — Identify primary industries
        yield {
            "type": "step_start",
            "step": 2,
            "title": "Identifying primary (first-order) affected industries & sectors",
            "detail": "Reasoning about direct causal mechanisms and financial line-item impacts...",
        }

        primary_prompt = (
            "Signal / development: " + macro_event + "\n\n"
            "Recent news / research context:\n" + combined_context[:3500] + "\n\n"
            "Task: Identify 4-6 PRIMARY industries or sectors DIRECTLY hit by this signal or development.\n\n"
            "For EACH industry provide:\n"
            "1. The exact causal mechanism.\n"
            "2. Which financial line item is affected.\n"
            "3. Impact direction + magnitude estimate.\n"
            "4. Typical reporting lag in quarters.\n"
            "5. Confidence level.\n\n"
            "Return ONLY this JSON schema:\n"
            "{\n"
            "  \"event_summary\": \"<one-sentence interpretation>\",\n"
            "  \"primary_industries\": [\n"
            "    {\n"
            "      \"industry\": \"<name>\",\n"
            "      \"mechanism\": \"<precise causal chain>\",\n"
            "      \"line_item_affected\": \"<specific P&L / balance-sheet line>\",\n"
            "      \"impact_direction\": \"negative | positive | mixed\",\n"
            "      \"impact_magnitude\": \"high | medium | low\",\n"
            "      \"lag_quarters\": \"0-1 | 1-2 | 2-4 | 4+\",\n"
            "      \"confidence\": \"high | medium | low\"\n"
            "    }\n"
            "  ]\n"
            "}"
        )

        primary_raw  = self._complete(primary_prompt, json_mode=True)
        primary_data = _safe_json(primary_raw, {"event_summary": "", "primary_industries": []})

        yield {"type": "step_result", "step": 2, "data": primary_data}

        # STEP 3 — Second-order / ripple effects
        yield {
            "type": "step_start",
            "step": 3,
            "title": "Mapping second-order ripple effects",
            "detail": "Tracing supply-chain, customer, substitution and financial-contagion linkages...",
        }

        p_names = [i["industry"] for i in primary_data.get("primary_industries", [])]
        q3 = macro_event + " supply chain spillover effects " + " ".join(p_names[:3])
        r3 = self._search(q3, max_results=4)

        yield {
            "type": "searches",
            "step": 3,
            "queries": [{"query": q3, "results": [r.get("title", "") for r in r3]}],
        }

        sc_context = self._fmt_results(r3)

        second_prompt = (
            "Signal / development: " + macro_event + "\n\n"
            "Primary industries and their direct impacts:\n"
            + json.dumps(primary_data.get("primary_industries", []), indent=2) + "\n\n"
            "Additional supply-chain / sector context:\n" + sc_context[:2500] + "\n\n"
            "Task: Identify 4-6 SECOND-ORDER industries affected because the primary industries are affected.\n"
            "Think about: upstream suppliers, downstream customers, substitution effects, financial contagion.\n\n"
            "Return ONLY this JSON schema:\n"
            "{\n"
            "  \"second_order_industries\": [\n"
            "    {\n"
            "      \"industry\": \"<name>\",\n"
            "      \"triggered_by\": \"<primary industry name>\",\n"
            "      \"causal_chain\": \"<step-by-step causal chain>\",\n"
            "      \"line_item_affected\": \"<specific line>\",\n"
            "      \"impact_direction\": \"negative | positive | mixed\",\n"
            "      \"impact_magnitude\": \"high | medium | low\",\n"
            "      \"lag_quarters\": \"1-2 | 2-4 | 4+\",\n"
            "      \"confidence\": \"high | medium | low\"\n"
            "    }\n"
            "  ]\n"
            "}"
        )

        second_raw  = self._complete(second_prompt, json_mode=True)
        second_data = _safe_json(second_raw, {"second_order_industries": []})

        yield {"type": "step_result", "step": 3, "data": second_data}

        # STEP 4 — Identify most-exposed listed companies
        yield {
            "type": "step_start",
            "step": 4,
            "title": "Identifying most-exposed listed companies",
            "detail": "Searching for specific stocks with highest exposure per sector...",
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
            q = "publicly listed companies most exposed to " + industry_name + " " + macro_event + " stocks earnings"
            results = self._search(q, max_results=3)
            search_log.append({"query": q, "results": [r.get("title", "") for r in results]})
            snippet = self._fmt_results(results)
            company_context_parts.append("### " + industry_name + " (" + order + ")\n" + snippet)

        yield {"type": "searches", "step": 4, "queries": search_log}

        company_context = "\n\n".join(company_context_parts)

        company_prompt = (
            "Signal / development: " + macro_event + "\n\n"
            "Industry impact summary:\n"
            "Primary:   " + json.dumps(primary_data.get("primary_industries", []), indent=2) + "\n"
            "Secondary: " + json.dumps(second_data.get("second_order_industries", []), indent=2) + "\n\n"
            "Company research context:\n" + company_context[:3500] + "\n\n"
            "Task: For EACH industry identify 2-3 listed companies with the highest financial exposure.\n\n"
            "Return ONLY this JSON schema:\n"
            "{\n"
            "  \"company_exposures\": [\n"
            "    {\n"
            "      \"industry\": \"<name>\",\n"
            "      \"order\": \"primary | secondary\",\n"
            "      \"companies\": [\n"
            "        {\n"
            "          \"name\": \"<Company Name>\",\n"
            "          \"ticker\": \"<TICK>\",\n"
            "          \"exchange\": \"<NYSE|NASDAQ|LSE|...>\",\n"
            "          \"why_exposed\": \"<specific business-model reason>\",\n"
            "          \"financial_impact\": \"<line item + estimated magnitude>\",\n"
            "          \"confirming_evidence\": \"<what to watch>\",\n"
            "          \"exposure_level\": \"high | medium\"\n"
            "        }\n"
            "      ]\n"
            "    }\n"
            "  ]\n"
            "}"
        )

        companies_raw  = self._complete(company_prompt, json_mode=True)
        companies_data = _safe_json(companies_raw, {"company_exposures": []})

        yield {"type": "step_result", "step": 4, "data": companies_data}

        # STEP 5 — Analyst synthesis
        yield {
            "type": "step_start",
            "step": 5,
            "title": "Synthesising analyst narrative",
            "detail": "Writing the final transmission map summary...",
        }

        synthesis_prompt = (
            "You have completed a full signal-to-stock transmission analysis for:\n"
            "\"" + macro_event + "\"\n\n"
            "Event summary: " + primary_data.get("event_summary", "") + "\n"
            "Primary industries: " + json.dumps(primary_data.get("primary_industries", []), indent=2) + "\n"
            "Secondary industries: " + json.dumps(second_data.get("second_order_industries", []), indent=2) + "\n"
            "Company exposures: " + json.dumps(companies_data.get("company_exposures", []), indent=2) + "\n\n"
            "Write a concise, analyst-quality synthesis in 3-4 tight paragraphs:\n\n"
            "Paragraph 1 — What this macro event REALLY means mechanically.\n"
            "Paragraph 2 — The single most important transmission pathway.\n"
            "Paragraph 3 — The 2-3 companies where the impact will be MOST VISIBLE.\n"
            "Paragraph 4 — Confirming or disconfirming evidence to track over the next 1-4 quarters.\n\n"
            "Write as a senior analyst. Use British English. Be direct. No hedging. No bullet points. No markdown."
        )

        synthesis = self._complete(synthesis_prompt, json_mode=False)

        # FINAL — Emit the complete transmission map
        yield {
            "type": "transmission_map",
            "event": macro_event,
            "event_summary": primary_data.get("event_summary", ""),
            "synthesis": synthesis,
            "primary_industries": primary_data.get("primary_industries", []),
            "second_order_industries": second_data.get("second_order_industries", []),
            "company_exposures": companies_data.get("company_exposures", []),
        }


def fetch_recent_macro_events():
    """
    Search Tavily for today's major macro economic events, then use Groq to
    distil the results into 5-6 clean, concise event descriptions suitable
    for dropping straight into the analyser input field.
    Raises on failure so the caller can log and serve fallback events.
    """
    if not TAVILY_API_KEY:
        raise ValueError("TAVILY_API_KEY environment variable is not set")
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY environment variable is not set")

    tavily = TavilyClient(api_key=TAVILY_API_KEY)
    groq   = Groq(api_key=GROQ_API_KEY)

    queries = [
        "major macroeconomic data release market moving news today 2026",
        "central bank decision inflation GDP PMI economic report this week",
        "industry disruption supply chain pricing power technology shift earnings impact 2026",
    ]

    snippets = []
    errors = []
    for q in queries:
        try:
            results = tavily.search(q, max_results=6, search_depth="advanced")
            for r in results.get("results", []):
                title   = r.get("title", "").strip()
                content = r.get("content", "")[:200].strip()
                if title:
                    snippets.append("- " + title + ": " + content)
        except Exception as e:
            errors.append(str(e))

    if not snippets:
        raise RuntimeError("Tavily returned no results. Errors: " + "; ".join(errors))

    context = "\n".join(snippets[:14])

    prompt = (
        "Below are recent financial news headlines and market developments. Extract the 5-6 most "
        "significant signals — these can be macro events (central bank decisions, CPI, GDP, PMI, NFP), "
        "commodity or supply shocks, industry shifts (pricing power changes, technology disruption, "
        "demand destruction), or geopolitical events with clear economic consequences.\n\n"
        "Format each as a crisp, factual statement of 8-15 words, e.g.:\n"
        "- \"US CPI rises to 3.4% YoY in April, above consensus\"\n"
        "- \"Fed holds rates at 5.25%, signals one cut in 2025\"\n"
        "- \"China manufacturing PMI falls to 48.8, six-month low\"\n\n"
        "Use British English spelling. Do not include editorial commentary.\n\n"
        "Headlines:\n" + context + "\n\n"
        "Return ONLY valid JSON:\n"
        "{\"events\": [\"event1\", \"event2\", \"event3\", \"event4\", \"event5\"]}"
    )

    resp = groq.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.1,
        max_tokens=512,
    )

    data = _safe_json(resp.choices[0].message.content, {"events": []})
    events = data.get("events", [])
    if not events:
        raise RuntimeError("Groq returned empty events list")
    return events
