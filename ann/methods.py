"""How every index is built, wrapped for exact re-ranking, and tuned at query time.

All methods come from Faiss so that differences reflect algorithms, not code quality.
Compressed indexes (LSH, PQ, IVF-PQ, IVF-RaBitQ) are cached on disk WITHOUT the raw
vectors; `wrap_refine` adds the exact re-ranking layer (IndexRefineFlat) on load, so
the cache stays small while memory accounting still counts the raw vectors.
"""
import faiss

REFINED = {"lsh", "ivfpq", "ivfrabitq", "pq_refine"}


def tag(method, p):
    """File-name-safe identifier of one build configuration, e.g. ivfpq_M32_nlist1024."""
    if not p:
        return method
    return method + "_" + "_".join(f"{k}{int(v) if isinstance(v, bool) else v}" for k, v in sorted(p.items()))


def describe(method, d, p):
    """Human-readable Faiss recipe, logged with every result."""
    if method == "flat":
        return "IndexFlatL2"
    if method == "lsh":
        return f"IndexLSH(d={d}, nbits={p['nbits']}, rotate=True, train_thresholds={p['train_thresholds']}) + RefineFlat"
    if method in ("pq", "pq_refine"):
        return f"PQ{p['M']}x8np" + (" + RefineFlat" if method == "pq_refine" else "")
    if method == "ivfpq":
        return f"IVF{p['nlist']},PQ{p['M']}np + RefineFlat"
    if method == "hnsw":
        return f"HNSW{p['M']} (efConstruction={p['efC']})"
    if method == "ivfrabitq":
        return f"RR{d},IVF{p['nlist']},{_rabitq_enc(p)} + RefineFlat"
    if method == "ivfflat":
        return f"IVF{p['nlist']},Flat"
    raise ValueError(method)


def _rabitq_enc(p):
    return "RaBitQ" if p["bits"] == 1 else f"RaBitQ{p['bits']}"


def build_base(method, d, p):
    """Return the untrained index without the re-ranking layer."""
    if method == "flat":
        return faiss.IndexFlatL2(d)
    if method == "lsh":
        # random rotation, then one sign bit per output coordinate; nbits may exceed d
        return faiss.IndexLSH(d, p["nbits"], True, p["train_thresholds"])
    if method in ("pq", "pq_refine"):
        # "np" skips polysemous training, which is slow and not used here
        return faiss.index_factory(d, f"PQ{p['M']}x8np")
    if method == "ivfpq":
        # "np": Faiss turns on slow polysemous training for IVF-PQ by default (17x slower build, unused here)
        return faiss.index_factory(d, f"IVF{p['nlist']},PQ{p['M']}np")
    if method == "hnsw":
        # index_factory already returns an IndexHNSWFlat; do not downcast a temporary here:
        # the new wrapper would point at an object freed with the old one
        index = faiss.index_factory(d, f"HNSW{p['M']}")
        index.hnsw.efConstruction = p["efC"]           # must be set before add()
        return index
    if method == "ivfrabitq":
        # RR = random rotation, which RaBitQ's error bound assumes
        return faiss.index_factory(d, f"RR{d},IVF{p['nlist']},{_rabitq_enc(p)}")
    if method == "ivfflat":
        return faiss.index_factory(d, f"IVF{p['nlist']},Flat")
    raise ValueError(method)


def wrap_refine(base, xb):
    """Exact re-ranking layer over a trained, filled base index. k_factor=1 means no re-ranking."""
    index = faiss.IndexRefineFlat(base, faiss.swig_ptr(xb))
    index.referenced_objects = [base]                  # keep the Python base object alive
    index.k_factor = 1
    return index


def unwrap(index, cls):
    """Walk through PreTransform / Refine wrappers until an index of type cls.

    Faiss raises AttributeError when a parameter is set on the wrong wrapper
    (for example nprobe on an IndexPreTransform), so always go through here.
    """
    idx = faiss.downcast_index(index)
    while not isinstance(idx, cls):
        if isinstance(idx, faiss.IndexPreTransform):
            idx = faiss.downcast_index(idx.index)
        elif isinstance(idx, faiss.IndexRefine):
            idx = faiss.downcast_index(idx.base_index)
        else:
            raise TypeError(f"{cls.__name__} not found in {type(index).__name__}")
    return idx


def set_search_params(index, p):
    if "nprobe" in p:
        unwrap(index, faiss.IndexIVF).nprobe = p["nprobe"]
    if "qb" in p:
        unwrap(index, faiss.IndexIVFRaBitQ).qb = p["qb"]
    if "efSearch" in p:
        unwrap(index, faiss.IndexHNSW).hnsw.efSearch = p["efSearch"]
    if "k_factor" in p:
        unwrap(index, faiss.IndexRefine).k_factor = p["k_factor"]
    if "sdc" in p:
        pq = unwrap(index, faiss.IndexPQ)
        if p["sdc"]:
            if pq.pq.sdc_table.size() == 0:
                pq.pq.compute_sdc_table()                  # SDC needs centroid-to-centroid tables
            pq.search_type = faiss.IndexPQ.ST_SDC
        else:
            pq.search_type = faiss.IndexPQ.ST_PQ           # asymmetric distance (ADC), the default
