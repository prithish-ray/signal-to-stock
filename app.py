"""
Macro-to-Company Transmission Tester
Flask backend with SSE streaming.
"""

import os
import json
from flask import Flask, render_template, request, Response, stream_with_context
from agent import MacroTransmissionAgent, fetch_recent_macro_events

app = Flask(__name__)

FALLBACK_EVENTS = [
    "US ISM Manufacturing falls to 46, weakest in two years",
    "OPEC cuts production by 500k bbl/day from next quarter",
    "Fed raises rates by 50bps unexpectedly at emergency meeting",
    "China manufacturing PMI drops to 48.5, third consecutive contraction",
    "US 10-year yield spikes to 5.5%, highest since 2007",
]


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/macro-news")
def macro_news():
    error_detail = None
    try:
        events = fetch_recent_macro_events()
        if events:
            return {"events": events, "source": "live"}
        error_detail = "fetch returned empty list"
    except Exception as e:
        error_detail = str(e)

    # Log to stdout so Render logs capture it
    print("macro-news fallback triggered:", error_detail, flush=True)
    return {"events": FALLBACK_EVENTS, "source": "fallback", "error": error_detail}


@app.route("/analyse", methods=["POST"])
def analyse():
    data = request.get_json(silent=True) or {}
    macro_event = data.get("event", "").strip()

    if not macro_event:
        return {"error": "No event provided"}, 400

    def generate():
        try:
            agent = MacroTransmissionAgent()
            for step in agent.analyse(macro_event):
                yield "data: " + json.dumps(step) + "\n\n"
        except Exception as e:
            yield "data: " + json.dumps({"type": "error", "message": str(e)}) + "\n\n"
        finally:
            yield "data: " + json.dumps({"type": "done"}) + "\n\n"

    return Response(
        stream_with_context(generate()),
        content_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
