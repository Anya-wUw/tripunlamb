# -*- coding: utf-8 -*-
"""
Build QA dataset from triplet CSVs (robust + batched SPARQL + progress + checkpoints).

Requirements:
  pip install pandas tqdm SPARQLWrapper requests beautifulsoup4 spacy flair
  python -m spacy download en_core_web_lg
"""

from __future__ import annotations
import os
import re
import sys
import csv
import glob
import json
import time
import math
import signal
import argparse
from typing import Iterable, List, Dict, Tuple, Optional

import pandas as pd
from tqdm.auto import tqdm
from SPARQLWrapper import SPARQLWrapper, JSON
import spacy
from flair.models import SequenceTagger
from flair.data import Sentence

# ===========================
# Config (tune if needed)
# ===========================
BATCH_LABELS = 120            # labels per VALUES block when resolving QIDs
BATCH_QIDS = 250              # QIDs per VALUES block when fetching popularity
MAX_QUERY_CHARS = 40_000      # max characters in VALUES list to avoid 400/414
SPARQL_TIMEOUT = 30           # seconds
SPARQL_RETRIES = 5            # attempts per query
INTER_BATCH_SLEEP = 0.05      # small sleep to avoid throttling
CHECKPOINT_EVERY_FILES = 3    # checkpoint after processing N input files
USER_AGENT = "UNLamb-qa-builder/0.2 (research; mailto:you@example.com)"


# ===========================
# Global state (for SIGINT)
# ===========================
_OK_ROWS: List[dict] = []
_MISSED_ROWS: List[dict] = []
_OUTPUT_DIR: Optional[str] = None


# ===========================
# Init NLP & SPARQL
# ===========================
def init_env():
    global sequence_tagger, spacy_en_core_web, sparql

    # Flair NER (English)
    sequence_tagger = SequenceTagger.load('ner')
    # spaCy (English, large)
    spacy_en_core_web = spacy.load("en_core_web_lg")

    # SPARQL client
    sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
    sparql.setReturnFormat(JSON)
    sparql.agent = USER_AGENT


# ===========================
# Original helpers (kept)
# ===========================
def identifier_conversion(entity, property=False):
    """Kept for compatibility (not used in batched path)."""
    if not property:  # item Q…
        query = f"""
            SELECT ?identifier WHERE {{
                ?identifier rdfs:label "{entity}"@en.
            }}
            """
    else:  # property P…
        query = f""" 
            SELECT ?identifier WHERE {{
                ?property rdf:type wikibase:Property .
                ?identifier rdfs:label "{entity}"@en. 
            }}
            """
    property_pattern = r'^P\d+'
    node_pattern = r'^Q\d+'
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    if "results" in results and "bindings" in results["results"]:
        if not property:
            for result in results["results"]["bindings"]:
                identifier = result["identifier"]["value"].split("/")[-1]
                if re.match(node_pattern, identifier):
                    return identifier
        else:
            for result in results["results"]["bindings"]:
                identifier = result["identifier"]["value"].split("/")[-1]
                if re.match(property_pattern, identifier):
                    return identifier
    return None


def generate_question(subject, relation, object, topic, query_subject=False):
    """
    Your original logic, kept verbatim (no global changes).
    """
    object_type1 = None
    object_type2 = None
    object_type = None
    discard_flag = False
    convert_dict1 = {
        "PER": "PERSON",
        "LOC": "GPE"
    }

    # method 1 — Flair
    sentence = Sentence(object)
    sequence_tagger.predict(sentence)
    entities = sentence.get_spans('ner')
    if entities:
        object_type1 = entities[0].tag
        if object_type1 == "PER" or object_type1 == "LOC":
            object_type1 = convert_dict1[object_type1]
        else:
            object_type1 = None

    # method 2 — spaCy
    object_doc = spacy_en_core_web(object)
    if object_doc.ents:
        object_type2 = object_doc.ents[0].label_

    if object_type1:        
        if object_type1 == object_type2:
            object_type = object_type1
        else:
            discard_flag = True
    else:
        if object_type2 != "GPE" and object_type2 != "PERSON":
            object_type = object_type2
        else:
            discard_flag = True
            
    if discard_flag:
        return None

    subject_doc = spacy_en_core_web(relation)
    if subject_doc[-1].tag_ == "IN" and subject_doc[0].tag_ not in ["VBN", "VBD", "VB", "VBZ"]:
        return None
        
    qa = {}
    qa["subject"] = subject
    qa["relation"] = relation
    qa["object"] = object

    object_to_interrogative = {
        "PERSON": "Who",
        "DATE": "When",
    }
    interrogative = object_to_interrogative.get(object_type, "What")

    if query_subject:
        tmp = subject
        subject = object
        object = tmp

    if subject_doc[0].tag_ == "VBN" and subject_doc[-1].tag_ == "IN" and all(token.tag_ not in ["NN", "NNP", "NNPS", "NNS"] for token in subject_doc[0:]):
        if not query_subject:
            qa["question"] = interrogative + " was " + subject + " " + relation + "?"
            qa["label"] = object
        else:
            if object_type != "PERSON":
                first_pair = next(iter(topic.items()), None)
                if first_pair and first_pair[1] != "revolution":
                    interrogative = "Which " + first_pair[1]
                else:
                    interrogative = "Which revolution or war"
            qa["question"] = interrogative + " was " + relation + " " + object + "?"
            qa["label"] = subject

    elif subject_doc[0].tag_ == "JJ" and subject_doc[-1].tag_ == "IN" and all(token.tag_ not in ["NN", "NNP", "NNPS", "NNS"] for token in subject_doc[0:]):
        if not query_subject:
            qa["question"] = interrogative + " is " + subject + " " + relation + "?"
            qa["label"] = object
        else:
            if object_type != "PERSON":
                first_pair = next(iter(topic.items()), None)
                if first_pair and first_pair[1] != "revolution":
                    interrogative = "Which " + first_pair[1]
                else:
                    interrogative = "Which revolution or war"
            qa["question"] = interrogative + " is " + " " + relation + " " + object + "?"
            qa["label"] = subject
            
    elif subject_doc[0].tag_ == "VBD" and subject_doc[-1].tag_ not in ["NN", "NNP", "NNPS", "NNS"]:
        if not query_subject:
            qa["question"] = interrogative + " did " + subject + " "
            for token in subject_doc:
                if token.tag_ == "VBD":
                    qa["question"] += token.lemma_ + " "
                else:
                    qa["question"] += token.text + " "
            qa["question"] = qa["question"][:-1] + "?"
            qa["label"] = object
        else:
            if object_type != "PERSON":
                first_pair = next(iter(topic.items()), None)
                if first_pair and first_pair[1] != "revolution":
                    interrogative = "Which " + first_pair[1]
                else:
                    interrogative = "Which revolution or war"
            qa["question"] = interrogative + " " + relation + " " + object + "?"
            qa["label"] = subject

    elif (subject_doc[0].tag_ == "VB" or subject_doc[0].tag_ == "VBZ") and subject_doc[-1].tag_ not in ["NN", "NNP", "NNPS", "NNS"]:
        if not query_subject:
            qa["question"] = interrogative + " does " + subject + " "
            for token in subject_doc:
                if token.tag_ == "VBZ":
                    qa["question"] += token.lemma_ + " "
                else:
                    qa["question"] += token.text + " "
            qa["question"] = qa["question"][:-1] + "?"
            qa["label"] = object
        else:
            if object_type != "PERSON":
                first_pair = next(iter(topic.items()), None)
                if first_pair and first_pair[1] != "revolution":
                    interrogative = "Which " + first_pair[1]
                else:
                    interrogative = "Which revolution or war"
            qa["question"] = interrogative + " " + relation + " " + object + "?"
            qa["label"] = subject

    elif (subject_doc[-1].tag_ == "NN" or subject_doc[-1].tag_ == "NNP") and subject_doc[0].tag_ not in ["VB", "VBZ", "VBD"]: 
        if not query_subject:
            qa["question"] = interrogative + " is the " + relation + " of " + subject + "?"
            qa["label"] = object
        else:
            first_pair = next(iter(topic.items()), None)
            if first_pair and first_pair[1] == "human":
                qa["question"] = interrogative + "se " + relation + " is " + object + "?"
            else:
                if first_pair and first_pair[1] != "revolution":
                    interrogative = "Which " + first_pair[1]
                else:
                    interrogative = "Which revolution or war"
                qa["question"] = interrogative + "'s " + relation + " is " + object + "?"
            qa["label"] = subject
    else:
        return None
    return qa


def safe_generate_qa(subject: str, relation: str, obj: str, topic=None):
    """Try forward; if None, try reverse; else None."""
    topic = topic or {}
    qa = generate_question(subject, relation, obj, topic, query_subject=False)
    if qa:
        return qa
    qa_rev = generate_question(subject, relation, obj, topic, query_subject=True)
    if qa_rev:
        return qa_rev
    return None


# ===========================
# SPARQL helpers (robust)
# ===========================
def _sparql_run(query: str):
    """SPARQL with timeout + retries."""
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    try:
        sparql.setTimeout(SPARQL_TIMEOUT)
    except Exception:
        pass
    delay = 1.0
    for attempt in range(1, SPARQL_RETRIES + 1):
        try:
            return sparql.query().convert()
        except Exception as e:
            if attempt == SPARQL_RETRIES:
                raise
            time.sleep(delay)
            delay = min(delay * 2, 15)


def _chunks(lst: List[str], n: int) -> Iterable[List[str]]:
    for i in range(0, len(lst), n):
        yield lst[i:i+n]


def _sparql_escape_literal(text: str) -> str:
    """
    Safe SPARQL literal with @en:
      - escape backslashes and double quotes
      - strip control chars and newlines
    """
    if text is None:
        text = ""
    s = str(text)
    s = s.replace("\\", "\\\\").replace('"', '\\"')
    s = s.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    return f"\"{s}\"@en"


def _pack_values_by_size(literals: List[str], max_chars: int) -> List[List[str]]:
    """Split VALUES list by approximate size to avoid too-long queries."""
    packs, cur, cur_len = [], [], 0
    for lit in literals:
        add_len = len(lit) + (1 if cur else 0)  # +space
        if cur and (cur_len + add_len) > max_chars:
            packs.append(cur)
            cur, cur_len = [lit], len(lit)
        else:
            cur.append(lit)
            cur_len += add_len
    if cur:
        packs.append(cur)
    return packs


def resolve_qids_batched(labels: List[str]) -> Dict[str, Optional[str]]:
    """
    Resolve english label/altLabel to QID in batches.
    Returns: dict lower(label) -> QID | None
    """
    mapping: Dict[str, Optional[str]] = {}
    canon = [x.strip() for x in labels if isinstance(x, str) and x.strip()]

    # unique (case-insensitive) preservation
    lower_to_orig: Dict[str, str] = {}
    for x in canon:
        lx = x.lower()
        if lx not in lower_to_orig:
            lower_to_orig[lx] = x
    todo = list(lower_to_orig.keys())

    for batch_lower in tqdm(list(_chunks(todo, BATCH_LABELS)), desc="Resolve QIDs (batched)", position=1, leave=False):
        literals = [_sparql_escape_literal(lower_to_orig[l]) for l in batch_lower]
        for vals_group in _pack_values_by_size(literals, MAX_QUERY_CHARS):
            vals = " ".join(vals_group)
            query = f"""
            SELECT ?lab ?item WHERE {{
              VALUES ?lab {{ {vals} }}
              ?item a wikibase:Item ;
                    rdfs:label|skos:altLabel ?lab .
            }}
            """
            # default None
            for l in batch_lower:
                mapping.setdefault(l, None)

            res = _sparql_run(query)
            for b in res.get("results", {}).get("bindings", []):
                lab = b["lab"]["value"].lower()
                qid = b["item"]["value"].rsplit("/", 1)[-1]
                if qid.startswith("Q"):
                    if lab in mapping and mapping[lab] is None:
                        mapping[lab] = qid

            time.sleep(INTER_BATCH_SLEEP)
    return mapping


def fetch_popularity_batched(qids: List[str]) -> Dict[str, Tuple[int, int, int]]:
    """
    For QIDs returns (sitelinks, identifiers, total) in batches.
    """
    pop: Dict[str, Tuple[int, int, int]] = {}
    uniq = [q for q in qids if isinstance(q, str) and q.startswith("Q")]
    uniq = list(dict.fromkeys(uniq))
    for batch in tqdm(list(_chunks(uniq, BATCH_QIDS)), desc="Fetch popularity (batched)", position=1, leave=False):
        vals = " ".join([f"wd:{q}" for q in batch])
        query = f"""
        SELECT ?item ?sitelinks ?identifiers WHERE {{
          VALUES ?item {{ {vals} }}
          ?item wikibase:sitelinks ?sitelinks ;
                wikibase:identifiers ?identifiers .
        }}
        """
        res = _sparql_run(query)
        for b in res.get("results", {}).get("bindings", []):
            qid = b["item"]["value"].rsplit("/", 1)[-1]
            s = int(float(b["sitelinks"]["value"]))
            i = int(float(b["identifiers"]["value"]))
            pop[qid] = (s, i, s + i)
        time.sleep(INTER_BATCH_SLEEP)
    # default zeros for misses
    for q in uniq:
        if q not in pop:
            pop[q] = (0, 0, 0)
    return pop


# ===========================
# Core pipeline
# ===========================
def _write_checkpoint(ok_rows: List[dict], missed_rows: List[dict], output_dir: str):
    """Write lightweight checkpoints (idempotent)."""
    ok_df = pd.DataFrame(ok_rows)
    miss_df = pd.DataFrame(missed_rows)

    if not ok_df.empty:
        ok_df = ok_df.drop_duplicates(subset=["subjectLabel", "relation", "objectLabel"], keep="first")
        ok_df = ok_df.drop_duplicates(subset=["question", "answer"], keep="first")
        ok_df.to_csv(os.path.join(output_dir, "_qa_dataset_checkpoint.csv"), index=False)
    if not miss_df.empty:
        miss_df.to_csv(os.path.join(output_dir, "_qa_missed_checkpoint.csv"), index=False)


def _sigint_handler(signum, frame):
    """On Ctrl-C save checkpoints and exit gracefully."""
    if _OUTPUT_DIR:
        _write_checkpoint(_OK_ROWS, _MISSED_ROWS, _OUTPUT_DIR)
        print(f"\n[INTERRUPTED] Checkpoints saved in: {_OUTPUT_DIR}")
    sys.exit(1)


def build_qa_from_triplets_dir(
    input_dir: str,
    output_dir: Optional[str] = None,
    out_ok_name: str = "qa_dataset.csv",
    out_missed_name: str = "qa_missed_triplets.csv",
    rate_limit_sec: float = 0.0,
):
    global _OK_ROWS, _MISSED_ROWS, _OUTPUT_DIR
    _OK_ROWS, _MISSED_ROWS = [], []
    input_dir = os.path.abspath(input_dir)
    output_dir = output_dir or os.path.join(input_dir, "_qa_outputs")
    _OUTPUT_DIR = output_dir
    os.makedirs(output_dir, exist_ok=True)

    # ============ PASS 0: scan CSVs ============
    csv_files = sorted(glob.glob(os.path.join(input_dir, "**", "*.csv"), recursive=True))
    if not csv_files:
        print("Нет CSV в папке:", input_dir)
        return

    all_rows: List[Tuple[str, str, str, str]] = []  # (source, subj, rel, obj)
    subj_pool, obj_pool = [], []
    with tqdm(csv_files, desc="Scan CSVs", position=0, leave=True) as p_scan:
        for fp in p_scan:
            try:
                df = pd.read_csv(fp)
            except Exception as e:
                tqdm.write(f"[WARN] skip {fp}: {e}")
                continue

            required = {"subjectLabel", "relation", "objectLabel"}
            if not required.issubset(df.columns):
                tqdm.write(f"[WARN] {os.path.basename(fp)}: required columns missing")
                continue

            source = os.path.basename(fp)
            for _, r in df.iterrows():
                s = str(r["subjectLabel"]).strip()
                rel = str(r["relation"]).strip()
                o = str(r["objectLabel"]).strip()
                if not s or not rel or not o:
                    _MISSED_ROWS.append({"source": source, "subjectLabel": s, "relation": rel, "objectLabel": o, "reason": "empty cell"})
                    continue
                all_rows.append((source, s, rel, o))
                subj_pool.append(s)
                obj_pool.append(o)

    if not all_rows:
        pd.DataFrame(_MISSED_ROWS).to_csv(os.path.join(output_dir, out_missed_name), index=False)
        print("[MISS] Only missed rows saved (no valid data).")
        return

    # ============ PASS 1: resolve all QIDs (batched) ============
    unique_labels = list({x.strip() for x in (subj_pool + obj_pool) if x and isinstance(x, str)})
    qid_map = resolve_qids_batched(unique_labels)  # lower(label) -> QID | None

    # ============ PASS 2: fetch popularity (batched) ============
    all_qids = [qid for qid in qid_map.values() if qid]
    pop_map = fetch_popularity_batched(all_qids)   # QID -> (sitelinks, identifiers, total)

    # ============ PASS 3: generate QA (no network in loop) ============
    files_done = 0
    cur_file: Optional[str] = None
    with tqdm(total=len(all_rows), desc="Generate QA", position=0, leave=True) as p_rows:
        for (source, subj, rel, obj) in all_rows:
            if cur_file != source:
                cur_file = source
                files_done += 1
                if CHECKPOINT_EVERY_FILES and files_done % CHECKPOINT_EVERY_FILES == 0:
                    _write_checkpoint(_OK_ROWS, _MISSED_ROWS, output_dir)
                    tqdm.write(f"[checkpoint] after files: {files_done}")

            try:
                qa = safe_generate_qa(subj, rel, obj, topic={})
            except Exception as e:
                _MISSED_ROWS.append({"source": source, "subjectLabel": subj, "relation": rel, "objectLabel": obj,
                                     "reason": f"generate_question error: {e}"})
                p_rows.update(1)
                continue

            if not qa or not qa.get("question") or not qa.get("label"):
                _MISSED_ROWS.append({"source": source, "subjectLabel": subj, "relation": rel, "objectLabel": obj,
                                     "reason": "no QA generated"})
                p_rows.update(1)
                continue

            s_qid = qid_map.get(subj.lower())
            o_qid = qid_map.get(obj.lower())
            s_sl, s_id, s_sum = pop_map.get(s_qid, (0, 0, 0)) if s_qid else (0, 0, 0)
            o_sl, o_id, o_sum = pop_map.get(o_qid, (0, 0, 0)) if o_qid else (0, 0, 0)

            _OK_ROWS.append({
                "source": source,
                "question": qa["question"],
                "answer": qa["label"],
                "subjectLabel": subj,
                "relation": rel,
                "objectLabel": obj,
                "subject_qid": s_qid,
                "object_qid": o_qid,
                "subject_sitelinks": s_sl,
                "subject_identifiers": s_id,
                "object_sitelinks": o_sl,
                "object_identifiers": o_id,
                "subject_popularity": s_sum,
                "object_popularity": o_sum,
                "popularity_sum": s_sum + o_sum,
            })

            p_rows.update(1)
            if rate_limit_sec > 0:
                time.sleep(rate_limit_sec)

    # ============ Finalize & dedupe ============
    ok_df = pd.DataFrame(_OK_ROWS)
    miss_df = pd.DataFrame(_MISSED_ROWS)

    if not ok_df.empty:
        ok_df = ok_df.drop_duplicates(subset=["subjectLabel", "relation", "objectLabel"], keep="first")
        ok_df = ok_df.drop_duplicates(subset=["question", "answer"], keep="first")

    out_ok = os.path.join(output_dir, out_ok_name)
    out_miss = os.path.join(output_dir, out_missed_name)
    ok_df.to_csv(out_ok, index=False)
    miss_df.to_csv(out_miss, index=False)

    print(f"[OK] Saved {len(ok_df)} rows to: {out_ok}")
    print(f"[MISS] Saved {len(miss_df)} rows to: {out_miss}")


# ===========================
# CLI
# ===========================
def parse_args():
    p = argparse.ArgumentParser(description="Build QA dataset from Wikidata triplets CSVs.")
    p.add_argument("--input_dir", type=str, required=False,
                   default="/mnt/extremessd10tb/borisiuk/open-unlearning/constuction_full_triplets_ds/hallu-edit-bench-unfilterd-triplets",
                   help="Directory with CSVs (recursive).")
    p.add_argument("--output_dir", type=str, default=None,
                   help="Where to save outputs. Default: <input_dir>/_qa_outputs")
    p.add_argument("--ok_name", type=str, default="qa_dataset.csv")
    p.add_argument("--miss_name", type=str, default="qa_missed_triplets.csv")
    p.add_argument("--rate_limit_sec", type=float, default=0.0)
    return p.parse_args()


if __name__ == "__main__":
    # graceful SIGINT (save checkpoints)
    signal.signal(signal.SIGINT, _sigint_handler)

    init_env()
    args = parse_args()
    build_qa_from_triplets_dir(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        out_ok_name=args.ok_name,
        out_missed_name=args.miss_name,
        rate_limit_sec=args.rate_limit_sec,
    )
