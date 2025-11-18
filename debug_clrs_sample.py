# scripts/debug_clrs_sample.py
import json
import numpy as np
import traceback

def safe_shape(x):
    try:
        import tensorflow as tf
        if 'tensorflow' in type(x).__module__:
            a = np.array(x)
            return a.shape, a.dtype
    except Exception:
        pass
    try:
        return np.shape(x), getattr(x, "dtype", type(x))
    except Exception:
        return str(type(x)), None

try:
    import clrs
    print("Imported clrs:", clrs.__file__)
    # create_dataset may return different things depending on version:
    res = clrs.create_dataset(folder="./data/tmp/", algorithm="bellman_ford", batch_size=1, split="train")
    print("create_dataset returned:", type(res))
    # handle possible signatures
    if isinstance(res, tuple) and len(res) >= 1:
        dataset = res[0]
        print("Dataset object type:", type(dataset))
        try:
            iterator = dataset.as_numpy_iterator()
            sample = next(iterator)
        except Exception:
            # some versions return (iterator, num_samples, spec)
            if hasattr(res[0], "__next__"):
                sample = next(res[0])
            else:
                raise
    else:
        raise RuntimeError("create_dataset returned unexpected type")

    print("\nTop-level keys on sample (dir):")
    print(dir(sample)[:50])

    # Try to print features and outputs safely
    print("\n--- features.inputs ---")
    try:
        for inp in sample.features.inputs:
            print("input:", inp.name, "shape/dtype:", safe_shape(inp.data))
    except Exception as e:
        print("Could not list features.inputs:", e)
        traceback.print_exc()

    print("\n--- features.hints (lengths) ---")
    try:
        print("lengths shape:", safe_shape(sample.features.lengths))
    except Exception:
        pass

    print("\n--- outputs ---")
    try:
        for out in sample.outputs:
            print("output:", out.name, "shape/dtype:", safe_shape(out.data))
    except Exception as e:
        print("Could not list outputs:", e)
        traceback.print_exc()

    # Attempt to find A, distances, pi
    def find_named(container, names):
        for item in container:
            if getattr(item, "name", None) in names:
                return item
        return None

    A = find_named(sample.features.inputs, ("A", "adj", "adjacency", "adj_matrix"))
    print("\nFound adjacency (A):", hasattr(A, "name") and A.name or None, safe_shape(A.data) if A else None)

    distances = find_named(sample.outputs, ("distances", "dist", "distance"))
    preds = find_named(sample.outputs, ("pi", "predecessors", "predecessor", "parents"))

    print("Found distances:", getattr(distances, "name", None), safe_shape(getattr(distances, "data", None)))
    print("Found predecessors:", getattr(preds, "name", None), safe_shape(getattr(preds, "data", None)))

except Exception as e:
    print("ERROR in debug script:", e)
    traceback.print_exc()
