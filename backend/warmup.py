"""Train and cache both datasets' models (run at build time so the server starts ready)."""

import time

from backend import model_store as store

if __name__ == "__main__":
    for ds in store.DATASETS:
        t = time.time()
        store.get_bundle(ds)
        print(f"ready: {ds} in {time.time() - t:.0f}s", flush=True)
