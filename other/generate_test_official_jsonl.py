#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import re
from pathlib import Path
from collections import defaultdict, Counter

OFFICIAL_JSON = "/home/yuqing/Datas/VQA-RAD/VQA_RAD_Dataset_Public.json"
CURRENT_TEST_JSONL = "/home/yuqing/Datas/VQA-RAD/test.jsonl"
OUTPUT_JSONL = "/home/yuqing/Datas/VQA-RAD/test_official.jsonl"


def normalize_text(text: str) -> str:
    if text is None:
        return ""
    text = str(text).strip().lower()

    # 去掉多余引号
    text = text.replace('“', '"').replace('”', '"').replace("’", "'").replace("‘", "'")

    # 把制表符和多空格统一掉
    text = text.replace("\t", " ")
    text = re.sub(r"\s+", " ", text)

    # 去掉句末标点和多余标点
    text = text.strip(" .,:;!?\"'")

    return text


def load_official_data(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    pair_index = defaultdict(list)
    q_only_index = defaultdict(list)

    for item in data:
        q = normalize_text(item.get("question", ""))
        a = normalize_text(item.get("answer", ""))
        pair_index[(q, a)].append(item)
        q_only_index[q].append(item)

    return data, pair_index, q_only_index


def load_jsonl(path: str):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            row["_line_idx"] = line_idx
            rows.append(row)
    return rows


def all_same_meta(candidates):
    if not candidates:
        return True
    keys = ["question_type", "answer_type", "phrase_type"]

    # 【修复点】：明确提取出第一个候选者
    first_candidate = candidates[0]
    first = tuple(str(first_candidate.get(k, "")) for k in keys)

    for c in candidates[1:]:
        cur = tuple(str(c.get(k, "")) for k in keys)
        if cur != first:
            return False
    return True


def pick_candidate(candidates, used_ids):
    """
    优先挑一个还没用过的官方样本。
    如果都用过了，就返回第一个。
    """
    for c in candidates:
        qid = str(c.get("qid", ""))
        if qid not in used_ids:
            return c
    return candidates[0] if candidates else None


def enrich_test_rows(current_rows, pair_index, q_only_index):
    used_qids = set()
    enriched = []

    stats = Counter()
    unmatched_rows = []
    ambiguous_rows = []

    for row in current_rows:
        q = normalize_text(row.get("question", ""))
        a = normalize_text(row.get("answer", ""))

        pair_candidates = pair_index.get((q, a), [])
        match_mode = None
        chosen = None

        if len(pair_candidates) == 1:
            chosen = pair_candidates[0]
            match_mode = "pair_exact_unique"
        elif len(pair_candidates) > 1:
            chosen = pick_candidate(pair_candidates, used_qids)
            match_mode = "pair_exact_multi"
            if not all_same_meta(pair_candidates):
                ambiguous_rows.append({
                    "line_idx": row["_line_idx"],
                    "question": row.get("question", ""),
                    "answer": row.get("answer", ""),
                    "num_candidates": len(pair_candidates),
                    "reason": "multiple official rows share same (question, answer) but metadata differs"
                })
        else:
            q_candidates = q_only_index.get(q, [])
            if len(q_candidates) == 1:
                chosen = q_candidates[0]
                match_mode = "question_only_unique"
            elif len(q_candidates) > 1:
                # 如果 question-only 多个候选，再尝试 answer 的弱匹配
                same_answer = [
                    c for c in q_candidates
                    if normalize_text(c.get("answer", "")) == a
                ]
                if len(same_answer) >= 1:
                    chosen = pick_candidate(same_answer, used_qids)
                    match_mode = "question_only_answer_filtered"
                else:
                    chosen = pick_candidate(q_candidates, used_qids)
                    match_mode = "question_only_multi_fallback"
                    ambiguous_rows.append({
                        "line_idx": row["_line_idx"],
                        "question": row.get("question", ""),
                        "answer": row.get("answer", ""),
                        "num_candidates": len(q_candidates),
                        "reason": "matched by question only with multiple candidates"
                    })
            else:
                unmatched_rows.append({
                    "line_idx": row["_line_idx"],
                    "question": row.get("question", ""),
                    "answer": row.get("answer", "")
                })
                match_mode = "unmatched"

        out = {
            "index": row["_line_idx"],
            "image_name": row.get("image_name", ""),
            "question": row.get("question", ""),
            "answer": row.get("answer", "")
        }

        if chosen is not None:
            qid = str(chosen.get("qid", ""))
            used_qids.add(qid)

            out.update({
                "qid": chosen.get("qid", ""),
                "official_image_name": chosen.get("image_name", ""),
                "phrase_type": chosen.get("phrase_type", ""),
                "qid_linked_id": chosen.get("qid_linked_id", ""),
                "image_case_url": chosen.get("image_case_url", ""),
                "image_organ": chosen.get("image_organ", ""),
                "evaluation": chosen.get("evaluation", ""),
                "question_rephrase": chosen.get("question_rephrase", ""),
                "question_relation": chosen.get("question_relation", ""),
                "question_frame": chosen.get("question_frame", ""),
                "question_type": chosen.get("question_type", ""),
                "answer_type": chosen.get("answer_type", "")
            })
        else:
            out.update({
                "qid": "",
                "official_image_name": "",
                "phrase_type": "",
                "qid_linked_id": "",
                "image_case_url": "",
                "image_organ": "",
                "evaluation": "",
                "question_rephrase": "",
                "question_relation": "",
                "question_frame": "",
                "question_type": "",
                "answer_type": ""
            })

        out["_match_mode"] = match_mode
        enriched.append(out)
        stats[match_mode] += 1

    return enriched, stats, unmatched_rows, ambiguous_rows


def save_jsonl(rows, path: str):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    official_data, pair_index, q_only_index = load_official_data(OFFICIAL_JSON)
    current_rows = load_jsonl(CURRENT_TEST_JSONL)

    enriched, stats, unmatched_rows, ambiguous_rows = enrich_test_rows(
        current_rows, pair_index, q_only_index
    )

    save_jsonl(enriched, OUTPUT_JSONL)

    print("=" * 70)
    print(f"Official rows loaded: {len(official_data)}")
    print(f"Current test rows loaded: {len(current_rows)}")
    print(f"Output written to: {OUTPUT_JSONL}")
    print("-" * 70)
    print("Match stats:")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    print("-" * 70)
    print(f"Unmatched rows: {len(unmatched_rows)}")
    print(f"Ambiguous rows: {len(ambiguous_rows)}")

    # 统计官方字段覆盖情况
    answer_type_counter = Counter()
    question_type_counter = Counter()

    for row in enriched:
        if row.get("answer_type"):
            answer_type_counter[row["answer_type"]] += 1
        if row.get("question_type"):
            question_type_counter[row["question_type"]] += 1

    print("-" * 70)
    print("answer_type distribution:")
    for k, v in answer_type_counter.items():
        print(f"  {k}: {v}")

    print("-" * 70)
    print("Top question_type distribution:")
    for k, v in question_type_counter.most_common(20):
        print(f"  {k}: {v}")

    # 保存日志，方便你人工检查
    unmatched_path = str(Path(OUTPUT_JSONL).with_suffix("")) + "_unmatched.json"
    ambiguous_path = str(Path(OUTPUT_JSONL).with_suffix("")) + "_ambiguous.json"

    with open(unmatched_path, "w", encoding="utf-8") as f:
        json.dump(unmatched_rows, f, ensure_ascii=False, indent=2)

    with open(ambiguous_path, "w", encoding="utf-8") as f:
        json.dump(ambiguous_rows, f, ensure_ascii=False, indent=2)

    print("-" * 70)
    print(f"Unmatched log: {unmatched_path}")
    print(f"Ambiguous log: {ambiguous_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()