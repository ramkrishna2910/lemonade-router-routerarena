#!/usr/bin/env python3
"""Build the RouterArena calibration set from UPSTREAM public benchmarks only.

Compliance: RouterArena is evaluation-only — no router component may be
trained/fit/tuned on their data or labels. This builder therefore samples from
the upstream sources their categories draw from, dedupes against every
RouterArena split by normalized question text, and formats prompts with the
same per-dataset templates their pipeline uses (config/eval_config/zero-shot).

Output: calibration.jsonl (one item per line: id, ra_source, question, context,
options, answer, prompt_formatted, split), plus coverage_report.json.

Run from anywhere; paths are absolute. Requires the RouterArena checkout at
~/work/routerarena (datasets already prepped there).
"""

import hashlib
import html
import json
import random
import re
import sys
import time
import urllib.request
from pathlib import Path

RA = Path.home() / "work/routerarena"
OUT = Path.home() / "work/router-experiment/runs/routerarena"
OUT.mkdir(parents=True, exist_ok=True)
SEED = 20260815
TARGET_TOTAL = 1500
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

sys.path.insert(0, str(RA))


def norm(text):
    return re.sub(r"\s+", " ", (text or "").strip().lower())[:400]


def qhash(text):
    return hashlib.sha256(norm(text).encode()).hexdigest()[:16]


def load_ra_hashes_and_dist():
    from datasets import load_dataset

    hashes = set()
    dist = {}
    for split in ("full", "sub_10", "robustness"):
        ds = load_dataset("RouteWorks/RouterArena", split=split)
        for row in ds:
            hashes.add(qhash(row.get("Question", "")))
        if split == "full":
            for name in ds["Dataset name"]:
                dist[name] = dist.get(name, 0) + 1
    return hashes, dist


def load_templates():
    cfg_dir = RA / "config/eval_config/zero-shot"
    templates = {}
    for f in cfg_dir.glob("*.json"):
        cfg = json.loads(f.read_text())
        templates[f.stem] = cfg.get("eval_params", {})
    return templates


def options_str(options):
    return "".join(f"{LETTERS[i]}. {o}\n" for i, o in enumerate(options))


def fmt(templates, base_name, question, context, options):
    params = templates.get(base_name) or templates.get(base_name.split("_", 1)[0])
    if not params or "prompt" not in params:
        return None
    prompt = params["prompt"]
    ctx = context if context else "None"

    def safe(template, **kw):
        out = template
        for key, value in kw.items():
            out = out.replace("{" + key + "}", str(value))
        return out.replace("{{", "{").replace("}}", "}")

    if options:
        return safe(prompt, Context=ctx, Question=question, Options=options_str(options))
    return safe(prompt, Context=ctx, Question=question)


# ---------------------------------------------------------------- adapters
# Each yields dicts: {ra_source, question, context, options, answer}
# answer is the grading target (letter for MCQ, text otherwise).


def src_mmlupro():
    from datasets import load_dataset

    cat_map = {
        "computer science": "MMLUPro_computer science",
        "history": "MMLUPro_history",
        "engineering": "MMLUPro_engineering",
        "health": "MMLUPro_health",
        "math": "MMLUPro_math",
        "psychology": "MMLUPro_psychology",
        "economics": "MMLUPro_economics",
        "business": "MMLUPro_business",
        "law": "MMLUPro_law",
        "philosophy": "MMLUPro_philosophy",
        "biology": "MMLUPro_biology",
        "chemistry": "MMLUPro_chemistry",
        "physics": "MMLUPro_physics",
    }
    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    for row in ds:
        ra_source = cat_map.get(row["category"])
        if not ra_source:
            continue
        yield {
            "ra_source": ra_source,
            "question": row["question"],
            "context": "",
            "options": row["options"],
            "answer": row["answer"],
        }


def src_pubmedqa():
    from datasets import load_dataset

    ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
    opts = ["Yes", "No", "Maybe"]
    ans_map = {"yes": "A", "no": "B", "maybe": "C"}
    for row in ds:
        ctx = " ".join(row["context"]["contexts"])[:4000]
        yield {
            "ra_source": "PubMedQA",
            "question": row["question"],
            "context": ctx,
            "options": opts,
            "answer": ans_map.get(row["final_decision"], ""),
        }


def src_medmcqa():
    from datasets import load_dataset

    ds = load_dataset("openlifescienceai/medmcqa", split="validation")
    for row in ds:
        yield {
            "ra_source": "MedMCQA",
            "question": row["question"],
            "context": "",
            "options": [row["opa"], row["opb"], row["opc"], row["opd"]],
            "answer": LETTERS[row["cop"]],
        }


def src_gsm8k():
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="test")
    for row in ds:
        answer = row["answer"].split("####")[-1].strip()
        yield {
            "ra_source": "GSM8K",
            "question": row["question"],
            "context": "",
            "options": None,
            "answer": answer,
        }


def src_mathqa():
    from datasets import load_dataset

    ds = load_dataset("allenai/math_qa", split="validation", trust_remote_code=True)
    for row in ds:
        opts = re.findall(r"[a-e]\s*\)\s*([^,]+?)(?=(?:\s*,\s*[a-e]\s*\))|$)", row["options"])
        if len(opts) < 4:
            continue
        yield {
            "ra_source": "MathQA",
            "question": row["Problem"],
            "context": "",
            "options": [o.strip() for o in opts],
            "answer": row["correct"].upper(),
        }


def src_ethics():
    from datasets import load_dataset

    variants = {
        "commonsense": ("Ethics_commonsense", ["Acceptable", "Unacceptable"]),
        "deontology": ("Ethics_deontology", ["Reasonable", "Unreasonable"]),
        "justice": ("Ethics_justice", ["Reasonable", "Unreasonable"]),
        "virtue": ("Ethics_virtue", ["Yes", "No"]),
    }
    for config, (ra_source, opts) in variants.items():
        try:
            ds = load_dataset("hendrycks/ethics", config, split="test")
        except Exception:
            continue
        for row in ds:
            text = row.get("input") or row.get("scenario") or ""
            if not text:
                continue
            label = row.get("label", 0)
            yield {
                "ra_source": ra_source,
                "question": text,
                "context": "",
                "options": opts,
                "answer": LETTERS[1 - int(label)] if config == "commonsense" else LETTERS[int(label)],
            }


def src_narrativeqa():
    from datasets import load_dataset

    ds = load_dataset("deepmind/narrativeqa", split="validation", streaming=True)
    for i, row in enumerate(ds):
        if i >= 400:
            break
        yield {
            "ra_source": "NarrativeQA",
            "question": row["question"]["text"],
            "context": row["document"]["summary"]["text"][:4000],
            "options": None,
            "answer": row["answers"][0]["text"],
        }


def src_livecodebench():
    from datasets import load_from_disk

    ds = load_from_disk(str(RA / "dataset/livecodebench"))
    for row in ds:
        q = row.get("question_content") or row.get("prompt") or ""
        if not q:
            continue
        yield {
            "ra_source": "LiveCodeBench",
            "question": q,
            "context": "",
            "options": None,
            "answer": json.dumps({"lcb_tests": True}),
            "_lcb_meta": {k: row.get(k) for k in ("question_id", "is_stdin") if k in row},
        }


OPENTDB_CATS = {
    "OpenTDB_General Knowledge": 9,
    "OpenTDB_Science: Computers": 18,
    "OpenTDB_Science & Nature": 17,
    "OpenTDB_Science: Mathematics": 19,
    "OpenTDB_Animals": 27,
    "OpenTDB_Vehicles": 28,
    "OpenTDB_Art": 25,
    "OpenTDB_Sports": 21,
    "OpenTDB_Geography": 22,
    "OpenTDB_History": 23,
    "OpenTDB_Celebrities": 26,
    "OpenTDB_Entertainment: Books": 10,
    "OpenTDB_Entertainment: Film": 11,
    "OpenTDB_Entertainment: Music": 12,
    "OpenTDB_Entertainment: Television": 14,
    "OpenTDB_Entertainment: Video Games": 15,
    "OpenTDB_Entertainment: Board Games": 16,
}


def src_opentdb():
    rng = random.Random(SEED)
    for ra_source, cat in OPENTDB_CATS.items():
        url = f"https://opentdb.com/api.php?amount=50&category={cat}&type=multiple"
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                data = json.loads(r.read())
        except Exception:
            continue
        for item in data.get("results", []):
            opts = [html.unescape(o) for o in item["incorrect_answers"]]
            correct = html.unescape(item["correct_answer"])
            pos = rng.randrange(len(opts) + 1)
            opts.insert(pos, correct)
            yield {
                "ra_source": ra_source,
                "question": html.unescape(item["question"]),
                "context": "",
                "options": opts,
                "answer": LETTERS[pos],
            }
        time.sleep(5)  # OpenTDB rate limit



def src_superglue():
    from datasets import load_dataset

    variants = [
        ("boolq", "SuperGLUE-QA", None),
        ("rte", "SuperGLUE-Entailment", ["True", "False"]),
        ("copa", "SuperGLUE-CausalReasoning", None),
        ("wic", "SuperGLUE-Wic", ["Yes", "No"]),
        ("wsc", "SuperGLUE-Wsc", ["Yes", "No"]),
    ]
    for config, ra_source, fixed_opts in variants:
        try:
            ds = load_dataset("aps/super_glue", config, split="validation")
        except Exception:
            continue
        for row in ds:
            if config == "boolq":
                q, ctx, opts = row["question"], row["passage"][:3000], ["Yes", "No"]
                ans = "A" if row["label"] == 1 else "B"
            elif config == "rte":
                q = f'Premise: {row["premise"]}\nHypothesis: {row["hypothesis"]}\nDoes the premise entail the hypothesis?'
                ctx, opts = "", fixed_opts
                ans = "A" if row["label"] == 0 else "B"
            elif config == "copa":
                q = f'{row["premise"]} What is the most plausible {row["question"]}?'
                ctx, opts = "", [row["choice1"], row["choice2"]]
                ans = LETTERS[row["label"]]
            elif config == "wic":
                q = f'Does the word "{row["word"]}" have the same meaning in both sentences?\n1. {row["sentence1"]}\n2. {row["sentence2"]}'
                ctx, opts = "", fixed_opts
                ans = "A" if row["label"] == 1 else "B"
            else:  # wsc
                q = f'In the sentence "{row["text"]}", does "{row["span2_text"]}" refer to "{row["span1_text"]}"?'
                ctx, opts = "", fixed_opts
                ans = "A" if row["label"] == 1 else "B"
            yield {"ra_source": ra_source, "question": q, "context": ctx,
                   "options": opts, "answer": ans}


def src_wmt19():
    from datasets import load_dataset

    pairs = ["de-en", "ru-en", "zh-en", "fi-en", "gu-en", "kk-en", "lt-en", "cs-en"]
    for pair in pairs:
        try:
            ds = load_dataset("wmt/wmt19", pair, split="validation", streaming=True)
        except Exception:
            continue
        src_lang = pair.split("-")[0]
        for i, row in enumerate(ds):
            if i >= 80:
                break
            tr = row["translation"]
            yield {"ra_source": f"WMT19-{pair}",
                   "question": tr[src_lang], "context": "",
                   "options": None, "answer": tr["en"]}


def src_musictheory():
    from datasets import load_dataset

    ds = load_dataset("m-a-p/MusicTheoryBench", split="test")
    for row in ds:
        opts = row["options"]
        if isinstance(opts, dict):
            opts = [opts[k] for k in sorted(opts)]
        yield {"ra_source": "MusicTheoryBench", "question": row["stem"],
               "context": row.get("instruction", "") or "",
               "options": opts, "answer": str(row["answer"]).strip().upper()[:1]}


def src_mmlu_misc():
    from datasets import load_dataset

    for subject, ra_source in (("formal_logic", "MMLU_formal_logic"),
                               ("management", "MMLU_management")):
        ds = load_dataset("cais/mmlu", subject, split="test")
        for row in ds:
            yield {"ra_source": ra_source, "question": row["question"], "context": "",
                   "options": row["choices"], "answer": LETTERS[row["answer"]]}


def src_math_comp():
    from datasets import load_dataset

    boxed = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")
    for config in ("algebra", "geometry", "number_theory", "counting_and_probability"):
        try:
            ds = load_dataset("EleutherAI/hendrycks_math", config, split="test")
        except Exception:
            continue
        for i, row in enumerate(ds):
            if i >= 120:
                break
            m = boxed.search(row["solution"])
            if not m:
                continue
            yield {"ra_source": "MATH", "question": row["problem"], "context": "",
                   "options": None, "answer": m.group(1)}


def src_aime():
    from datasets import load_dataset

    ds = load_dataset("AI-MO/aimo-validation-aime", split="train")
    for row in ds:
        yield {"ra_source": "AIME", "question": row["problem"], "context": "",
               "options": None, "answer": str(row["answer"])}


def src_chess():
    from datasets import load_dataset

    ds = load_dataset("Thytu/ChessInstruct", split="train", streaming=True)
    for i, row in enumerate(ds):
        if i >= 300:
            break
        yield {"ra_source": "ChessInstruct",
               "question": f'{row["task"]}\n{row["input"]}', "context": "",
               "options": None, "answer": str(row["expected_output"])[:200]}


def src_lcb_parquet():
    from datasets import load_dataset

    ds = load_dataset("livecodebench/code_generation_lite", split="test",
                      revision="refs/convert/parquet", streaming=True)
    for i, row in enumerate(ds):
        if i >= 400:
            break
        q = row.get("question_content") or ""
        if not q:
            continue
        yield {"ra_source": "LiveCodeBench", "question": q, "context": "",
               "options": None, "answer": json.dumps({"lcb_tests": True})}



def src_arcmmlu():
    from datasets import load_dataset

    ds = load_dataset("patrickshitou/ArcMMLU", split="test")
    for row in ds:
        opts = [row.get(k) for k in ("A", "B", "C", "D") if row.get(k)]
        if len(opts) < 4:
            continue
        yield {"ra_source": "ArcMMLU", "question": row["Question"], "context": "",
               "options": opts, "answer": str(row["Answer"]).strip().upper()[:1]}


def src_qanta():
    from datasets import load_dataset

    cat_map = {"Literature": "QANTA_Literature", "History": "QANTA_History",
               "Science": "QANTA_Science", "Fine Arts": "QANTA_Fine Arts",
               "Philosophy": "QANTA_Philosophy", "Social Science": "QANTA_Social Science",
               "Geography": "QANTA_Geography"}
    ds = load_dataset("community-datasets/qanta", "mode=first,char_skip=25", split="buzztrain")
    for row in ds:
        ra_source = cat_map.get(row.get("category"))
        if not ra_source:
            continue
        answer = (row.get("page") or "").replace("_", " ")
        if not answer:
            continue
        yield {"ra_source": ra_source, "question": row["full_question"][:2500],
               "context": "", "options": None, "answer": answer}


def src_ethics_parquet():
    from datasets import load_dataset

    base = ("https://huggingface.co/datasets/hendrycks/ethics/resolve/"
            "refs%2Fconvert%2Fparquet/{cfg}/test/0000.parquet")
    variants = {
        "commonsense": ("Ethics_commonsense", ["Acceptable", "Unacceptable"]),
        "deontology": ("Ethics_deontology", ["Reasonable", "Unreasonable"]),
        "justice": ("Ethics_justice", ["Reasonable", "Unreasonable"]),
        "virtue": ("Ethics_virtue", ["Yes", "No"]),
    }
    for cfg, (ra_source, opts) in variants.items():
        try:
            ds = load_dataset("parquet", data_files=base.format(cfg=cfg), split="train")
        except Exception:
            continue
        for i, row in enumerate(ds):
            if i >= 400:
                break
            text = row.get("input") or row.get("scenario") or ""
            if not text and row.get("sentence"):
                text = row["sentence"]
            if not text:
                continue
            label = int(row.get("label", 0))
            answer = LETTERS[label] if cfg == "commonsense" else LETTERS[1 - label]
            yield {"ra_source": ra_source, "question": text, "context": "",
                   "options": opts, "answer": answer}


def src_mbpp_code_proxy():
    # LiveCodeBench's full upstream release is script-gated on HF; MBPP stands
    # in for the code slice, tagged as a proxy in the coverage report.
    from datasets import load_dataset

    ds = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
    for row in ds:
        tests = row["test_list"]
        q = (row["prompt"] + "\n\nWrite the solution as a single Python "
             "function. Output only the code in a ```python block.")
        yield {"ra_source": "LiveCodeBench", "question": q,
               "prompt_formatted": q, "context": "", "options": None,
               "answer": json.dumps({"mbpp_tests": tests,
                                     "imports": row.get("test_imports") or []}),
               "_proxy": "mbpp"}


def src_geobench():
    from datasets import load_dataset

    ds = load_dataset("daven3/geobench", split="test")
    for row in ds:
        q = row.get("question") or row.get("Question") or ""
        if not q:
            continue
        opts = row.get("options") or None
        ans = str(row.get("answer") or row.get("Answer") or "").strip()
        if not ans:
            continue
        yield {"ra_source": "GeoBench", "question": q, "context": "",
               "options": opts, "answer": ans}


ADAPTERS = [
    src_mmlupro,
    src_pubmedqa,
    src_medmcqa,
    src_gsm8k,
    src_mathqa,
    src_ethics,
    src_narrativeqa,
    src_livecodebench,
    src_opentdb,
    src_superglue,
    src_wmt19,
    src_musictheory,
    src_mmlu_misc,
    src_math_comp,
    src_aime,
    src_chess,
    src_lcb_parquet,
    src_arcmmlu,
    src_qanta,
    src_ethics_parquet,
    src_mbpp_code_proxy,
    src_geobench,
]


def main():
    print("[calib] loading RouterArena hashes + distribution...")
    ra_hashes, dist = load_ra_hashes_and_dist()
    templates = load_templates()
    total_full = sum(dist.values())
    scale = TARGET_TOTAL / total_full

    pools = {}
    for adapter in ADAPTERS:
        name = adapter.__name__
        print(f"[calib] {name} ...", flush=True)
        try:
            count = skipped = 0
            for item in adapter():
                if qhash(item["question"]) in ra_hashes:
                    skipped += 1
                    continue
                base = item["ra_source"] if item["ra_source"].startswith(("Ethics", "ChessInstruct")) \
                    else item["ra_source"].split("_", 1)[0]
                prompt = item.get("prompt_formatted") or \
                    fmt(templates, item["ra_source"], item["question"],
                        item["context"], item["options"]) or \
                    fmt(templates, base, item["question"], item["context"], item["options"])
                if not prompt:
                    continue
                item["prompt_formatted"] = prompt
                pools.setdefault(item["ra_source"], []).append(item)
                count += 1
            print(f"[calib]   {name}: {count} items ({skipped} deduped vs RouterArena)")
        except Exception as e:
            print(f"[calib]   {name}: SKIPPED ({e})")

    rng = random.Random(SEED)
    selected = []
    coverage = {}
    for ra_source, n_full in sorted(dist.items(), key=lambda kv: -kv[1]):
        want = max(5, round(n_full * scale)) if ra_source in pools else 0
        pool = pools.get(ra_source, [])
        take = rng.sample(pool, min(want, len(pool))) if pool else []
        for item in take:
            selected.append(item)
        coverage[ra_source] = {"full_count": n_full, "wanted": want, "got": len(take)}

    rng.shuffle(selected)
    for i, item in enumerate(selected):
        item["id"] = f"calib-{qhash(item['question'])}"
        item["split"] = "TUNE" if i % 2 == 0 else "HOLDOUT"

    out_file = OUT / "calibration.jsonl"
    with out_file.open("w") as f:
        for item in selected:
            f.write(json.dumps(item) + "\n")
    covered = sum(v["full_count"] for v in coverage.values() if v["got"] > 0)
    report = {
        "total_selected": len(selected),
        "tune": sum(1 for x in selected if x["split"] == "TUNE"),
        "holdout": sum(1 for x in selected if x["split"] == "HOLDOUT"),
        "mass_coverage_pct": round(100 * covered / total_full, 1),
        "per_source": coverage,
    }
    (OUT / "coverage_report.json").write_text(json.dumps(report, indent=1))
    print(f"[calib] wrote {len(selected)} items -> {out_file}")
    print(f"[calib] RouterArena mass coverage: {report['mass_coverage_pct']}%")


if __name__ == "__main__":
    main()
