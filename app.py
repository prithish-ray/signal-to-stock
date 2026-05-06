"""
Macro-to-Company Transmission Tester
Flask backend with SSE streaming.
"""

import os
import json
from flask import Flask, render_template, request, Response, stream_with_context
from agent import MacroTransmissionAgent, fetch_recent_macro_events

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/macro-news")
def macro_news():
    try:
        events = fetch_recent_macro_events()
        return {"events": events}
    except Exception as e:
        return {"events": [], "error": str(e)}, 500


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
