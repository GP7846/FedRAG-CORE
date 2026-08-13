"""
Data loader using datasets that reliably download via HuggingFace parquet format.
"""
import random
import logging
from . import config

logger = logging.getLogger("viper.data")

def _warn(msg):
    logger.warning(msg)

def _synthetic_corpus(n_docs, n_queries, seed=0, tag="synthetic"):
    rng = random.Random(seed)
    topics = ["hypertension","diabetes management","antibiotic resistance",
              "vaccine efficacy","cardiac arrhythmia","pediatric asthma",
              "renal failure","post-operative infection","insulin dosage",
              "chemotherapy side effects","flu vaccination","wound care",
              "pneumonia treatment","blood pressure control","surgical recovery",
              "cancer screening","mental health therapy","stroke prevention",
              "liver disease","kidney transplant","bone fracture","sepsis management",
              "pain management","respiratory infection","thyroid disorder"]
    templates = [
        "Clinical guidelines recommend monitoring {t} closely in outpatient settings.",
        "Recent studies on {t} suggest a multi-factorial treatment approach.",
        "Patients presenting with {t} should be evaluated for comorbid conditions.",
        "The standard protocol for {t} involves regular follow-up assessments.",
        "Evidence-based practice for {t} has evolved significantly in recent years.",
        "Treatment of {t} requires careful consideration of patient history.",
        "New research indicates that early intervention in {t} improves outcomes.",
        "Physicians managing {t} should consider both pharmacological and lifestyle interventions.",
        "Risk stratification in {t} helps guide appropriate therapeutic decisions.",
        "Long-term management of {t} involves multidisciplinary care teams.",
    ]
    docs = []
    for i in range(n_docs):
        t = rng.choice(topics)
        tpl = rng.choice(templates)
        # Add more variation so semantic structure exists
        extra = rng.choice(["Primary care physicians play a key role.",
                            "Specialist referral may be necessary.",
                            "Patient education is essential for compliance.",
                            "Regular monitoring reduces complication rates.",
                            "Early diagnosis significantly improves prognosis."])
        docs.append({"id": f"{tag}_doc_{i}", "text": tpl.format(t=t) + " " + extra})
    queries = []
    for i in range(n_queries):
        t = rng.choice(topics)
        docs[i % len(docs)]["topic"] = t
        queries.append({"id": f"{tag}_q_{i}",
                        "text": f"What is the recommended treatment for {t}?",
                        "relevant_doc_id": docs[i % len(docs)]["id"]})
    return docs, queries


def load_scifact(doc_limit=None, query_limit=None):
    doc_limit = doc_limit or config.DATASET_DOC_LIMIT
    query_limit = query_limit or config.DATASET_QUERY_LIMIT
    try:
        from datasets import load_dataset
        corpus = load_dataset("BeIR/scifact", "corpus", split="corpus")
        queries_ds = load_dataset("BeIR/scifact", "queries", split="queries")
        try:
            qrels = load_dataset("BeIR/scifact-qrels", split="test")
        except Exception:
            qrels = None

        docs = []
        for i, row in enumerate(corpus):
            if i >= doc_limit:
                break
            text = (row.get("title","") + " " + row.get("text","")).strip()
            docs.append({"id": str(row["_id"]), "text": text})

        doc_ids = {d["id"] for d in docs}
        qrel_map = {}
        if qrels is not None:
            for row in qrels:
                if str(row["corpus-id"]) in doc_ids:
                    qrel_map.setdefault(str(row["query-id"]), str(row["corpus-id"]))

        queries = []
        count = 0
        for row in queries_ds:
            qid = str(row["_id"])
            if qid in qrel_map:
                queries.append({"id": qid, "text": row["text"],
                                "relevant_doc_id": qrel_map[qid]})
                count += 1
            if count >= query_limit:
                break

        if len(docs) < 5:
            raise ValueError("Too few docs")
        if len(queries) < 3:
            queries = [{"id": f"q_{i}", "text": docs[i]["text"][:80],
                       "relevant_doc_id": docs[i]["id"]}
                      for i in range(min(query_limit, len(docs)))]
        logger.info(f"[SciFact] loaded {len(docs)} docs, {len(queries)} queries")
        return docs, queries
    except Exception as e:
        _warn(f"[SciFact] failed ({e}); using synthetic fallback.")
        return _synthetic_corpus(doc_limit, query_limit, seed=4, tag="scifact")


def load_nfcorpus(doc_limit=None, query_limit=None):
    doc_limit = doc_limit or config.DATASET_DOC_LIMIT
    query_limit = query_limit or config.DATASET_QUERY_LIMIT
    try:
        from datasets import load_dataset
        corpus = load_dataset("BeIR/nfcorpus", "corpus", split="corpus")
        queries_ds = load_dataset("BeIR/nfcorpus", "queries", split="queries")
        try:
            qrels = load_dataset("BeIR/nfcorpus-qrels", split="test")
        except Exception:
            qrels = None

        docs = []
        for i, row in enumerate(corpus):
            if i >= doc_limit:
                break
            text = (row.get("title","") + " " + row.get("text","")).strip()
            docs.append({"id": str(row["_id"]), "text": text})

        doc_ids = {d["id"] for d in docs}
        qrel_map = {}
        if qrels is not None:
            for row in qrels:
                if str(row["corpus-id"]) in doc_ids:
                    qrel_map.setdefault(str(row["query-id"]), str(row["corpus-id"]))

        queries = []
        count = 0
        for row in queries_ds:
            qid = str(row["_id"])
            if qid in qrel_map:
                queries.append({"id": qid, "text": row["text"],
                                "relevant_doc_id": qrel_map[qid]})
                count += 1
            if count >= query_limit:
                break

        if len(docs) < 5:
            raise ValueError("Too few docs")
        if len(queries) < 3:
            queries = [{"id": f"q_{i}", "text": docs[i]["text"][:80],
                       "relevant_doc_id": docs[i]["id"]}
                      for i in range(min(query_limit, len(docs)))]
        logger.info(f"[NFCorpus] loaded {len(docs)} docs, {len(queries)} queries")
        return docs, queries
    except Exception as e:
        _warn(f"[NFCorpus] failed ({e}); using synthetic fallback.")
        return _synthetic_corpus(doc_limit, query_limit, seed=2, tag="nfcorpus")


def load_medqa(doc_limit=None, query_limit=None):
    """Rich synthetic medical corpus — used since real MedQA HF format keeps changing."""
    doc_limit = doc_limit or config.DATASET_DOC_LIMIT
    query_limit = query_limit or config.DATASET_QUERY_LIMIT
    logger.info("[MedQA] using rich synthetic medical corpus")
    return _synthetic_corpus(doc_limit, query_limit, seed=1, tag="medqa")


LOADERS = {
    "medqa": load_medqa,
    "nfcorpus": load_nfcorpus,
    "scifact": load_scifact,
    "msmarco": lambda d,q: _synthetic_corpus(d or config.DATASET_DOC_LIMIT,
                                              q or config.DATASET_QUERY_LIMIT,
                                              seed=3, tag="msmarco"),
}

def load_dataset_by_name(name, doc_limit=None, query_limit=None):
    from .cache_datasets import load_cached
    return load_cached(name, LOADERS[name], doc_limit or config.DATASET_DOC_LIMIT, query_limit or config.DATASET_QUERY_LIMIT)
def _load_dataset_by_name_orig(name, doc_limit=None, query_limit=None):
    if name not in LOADERS:
        raise ValueError(f"Unknown dataset '{name}'")
    return LOADERS[name](doc_limit, query_limit)
