"""
Macro-to-Company Transmission Tester
Flask backend — serves the UI and streams SSE events from the agent.
"""

import os
import json
from flask import Flask, render_template, request, Response, stream_with_context

from agent import MacroTransmissionAgent

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    data        = request.get_json(silent=True) or {}
    macro_event = data.get("event", "").strip()

    if not macro_event:
        return {"error": "No event provided"}, 400

    def generate():
        try:
            agent = MacroTransmissionAgent()
            for step in agent.analyze(macro_event):
                yield f"data: {json.dumps(step)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
        finally:
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return Response(
        stream_with_context(generate()),
        content_type="text/event-stream",
        headers={
            "Cache-Control":    "no-cache",
            "X-Accel-Buffering": "no",
            "Connection":       "keep-alive",
        },
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
