"""Live end-to-end test of the deployed HF Space, driven exactly like the demo.

Test 5 is the one that matters: same query, grader ON vs OFF, are the answers
different? That is the entire premise of the project.
"""
import json
import time

from gradio_client import Client

SPACE = "SallahAhmed/corrective-rag-relevance-grader"
client = Client(SPACE, verbose=False)

results = {}
for corpus in ["colab-t4", "espresso", "amsterdam"]:
    try:
        default_query = client.predict(corpus_key=corpus, api_name="/_default_query")
    except Exception as exc:
        default_query = f"<failed: {type(exc).__name__}>"
    print(f"[{time.strftime('%H:%M:%S')}] {corpus}: default query = {default_query!r}", flush=True)
    results[corpus] = {"default_query": default_query}

# The headline test: grader ON vs OFF on the same corpus/query.
CORPUS = "espresso"
QUERY = results["espresso"]["default_query"]
if QUERY.startswith("<failed"):
    QUERY = "why does my espresso taste sour"

arms = {}
for use_grader in (False, True):
    label = "ON " if use_grader else "OFF"
    t0 = time.time()
    try:
        out = client.predict(
            corpus_key=CORPUS, query=QUERY, top_k=4, max_new_tokens=180,
            use_grader=use_grader, api_name="/run_query",
        )
        dt = time.time() - t0
        print(f"[{time.strftime('%H:%M:%S')}] grader {label}: {dt:.1f}s", flush=True)
        arms[use_grader] = {"seconds": round(dt, 1), "output": out}
    except Exception as exc:
        dt = time.time() - t0
        print(f"[{time.strftime('%H:%M:%S')}] grader {label}: FAILED after {dt:.1f}s "
              f"-- {type(exc).__name__}: {exc}", flush=True)
        arms[use_grader] = {"seconds": round(dt, 1), "error": f"{type(exc).__name__}: {exc}"}

print("\n" + "=" * 70)
print("ARMS")
print("=" * 70)
for k, v in arms.items():
    print(f"\n--- grader {'ON' if k else 'OFF'} ({v['seconds']}s) ---")
    if "error" in v:
        print("  ERROR:", v["error"])
        continue
    out = v["output"]
    if isinstance(out, (list, tuple)):
        for i, part in enumerate(out):
            print(f"  [{i}] {str(part)[:900]}")
    else:
        print(str(out)[:1500])

# Did the two arms actually differ?
print("\n" + "=" * 70)
print("VERDICT")
print("=" * 70)
off, on = arms.get(False), arms.get(True)
if off and on and "output" in off and "output" in on:
    same = json.dumps(off["output"], default=str) == json.dumps(on["output"], default=str)
    print("ARMS IDENTICAL:", same, "->", "TEST 5 FAIL (no visible difference)"
          if same else "TEST 5 PASS (grader changes the output)")
    blob = json.dumps(on["output"], default=str)
    print("adapter mentioned in ON arm:", "adapter" in blob.lower())
    print("zero-shot banner present:", "zero-shot" in blob.lower())
else:
    print("Could not compare arms:", {k: list(v.keys()) for k, v in arms.items()})

with open("logs/space_live_test.json", "w", encoding="utf-8") as fh:
    json.dump({"corpora": results, "arms": arms}, fh, indent=2, default=str)
print("\nwrote logs/space_live_test.json")