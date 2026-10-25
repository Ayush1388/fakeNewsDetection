"""Train and cache both datasets' models (run at build time so the server starts ready)."""

from backend import model_store as store

if __name__ == "__main__":
    for ds in store.DATASETS:
        store.get_bundle(ds)
        print("ready:", ds)
