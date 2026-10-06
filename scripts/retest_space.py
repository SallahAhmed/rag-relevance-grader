import os
import time

from gradio_client import Client
from huggingface_hub import HfApi

SPACE = "SallahAhmed/corrective-rag-relevance-grader"
TOKEN = os.environ["HF_TOKEN"]
api = HfApi(token=TOKEN)

stage = None
for i in range(40):
    try:
        info = api.space_info(SPACE)
        s = f"{info.runtime.stage}/{getattr(info.runtime, 'hardware', None)}"
        if s != stage:
            print(f"[{time.strftime('%H:%M:%S')}] stage = {s}", flush=True)
            stage = s
        if info.runtime.stage == "RUNNING":
            print("SPACE RUNNING", flush=True)
            break
    except Exception as e:
        print(f"[{time.strftime('%H:%M:%S')}] info err {type(e).__name__}", flush=True)
    time.sleep(20)
else:
    print("never reached RUNNING", flush=True)

time.sleep(10)

c = None
for attempt in range(5):
    try:
        c = Client(SPACE, verbose=False)
        print(f"[{time.strftime('%H:%M:%S')}] client connected (attempt {attempt+1})", flush=True)
        break
    except Exception as e:
        print(f"[{time.strftime('%H:%M:%S')}] connect {attempt+1} {type(e).__name__}", flush=True)
        time.sleep(25)
if c is None:
    raise SystemExit(1)

q = c.predict(corpus_key="espresso", api_name="/_default_query")
print("query:", q, flush=True)
arms = {}
for ug in (False, True):
    t0 = time.time()
    out = c.predict(corpus_key="espresso", query=q, top_k=4,
                    max_new_tokens=160, use_grader=ug, api_name="/run_query")
    dt = time.time() - t0
    arms[ug] = out
    print(f"[{time.strftime('%H:%M:%S')}] grader {'ON ' if ug else 'OFF'}: {dt:.1f}s", flush=True)
    for i, part in enumerate(out):
        s = str(part)
        if i < 2:
            print(f"    [{i}] {s[:260]}", flush=True)

same = all(str(arms[False][i]) == str(arms[True][i]) for i in range(len(arms[False])))
print("ARMS_IDENTICAL:", same, flush=True)
blob = " ".join(str(x) for x in arms[True]).lower()
print("mentions_replay:", "replay" in blob, flush=True)
print("mentions_adapter:", "adapter" in blob, flush=True)
print("TEST_COMPLETE", flush=True)