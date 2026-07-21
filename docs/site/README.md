# Docs site

Professional landing page for MagicMindNet (HTML/CSS + Chart.js).

Open locally:

```bash
# from repo root
python -m http.server 8765 --directory docs/site
# then visit http://127.0.0.1:8765/
```

Or open `docs/site/index.html` directly (charts need a local HTTP server or `file:` may block `fetch` of JSON — use the http.server command above).

## Data

| File | Source |
|------|--------|
| `data/interop_benchmark.json` | `python examples/interop_benchmark.py` (sizes / save / load) |
| `data/smoke_eval.json` | `python -m magicmindnet.eval smoke --json -o …` |

Refresh after meaningful perf changes and commit the JSON so charts stay reproducible offline.
