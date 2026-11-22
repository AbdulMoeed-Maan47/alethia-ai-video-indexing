"""Compare a candidate results file with a reference results file.

Usage: python compare.py ref.jsonl candidate.jsonl

Prints the three metrics from the brief and PASS/FAIL for each:
  1. object F1            >= 0.95
  2. people exact match   >= 85% of seconds
  3. people within +/-1   >= 95% of seconds
"""
import json
import sys

OBJECTS = ["person", "car", "dog", "laptop", "cell phone"]


def load(path):
    """Return ({(video, second): (set_of_objects, people)}, error_count)."""
    rows = {}
    errors = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if "error" in r:
                errors += 1
                continue
            rows[(r["video"], r["second"])] = (set(r["objects"]), r["people"])
    return rows, errors


def main():
    ref, ref_err = load(sys.argv[1])
    cand, cand_err = load(sys.argv[2])

    # Every (video, second) that appears in either file is scored.
    # A second missing from one file counts as empty ([] and 0) there.
    keys = sorted(set(ref) | set(cand))
    empty = (set(), 0)
    only_ref = sum(1 for k in keys if k not in cand)
    only_cand = sum(1 for k in keys if k not in ref)

    # Object F1: treat every (second, class) pair as a yes/no decision.
    tp = {o: 0 for o in OBJECTS}
    fp = {o: 0 for o in OBJECTS}
    fn = {o: 0 for o in OBJECTS}
    exact = 0
    within1 = 0
    for k in keys:
        r_objs, r_people = ref.get(k, empty)
        c_objs, c_people = cand.get(k, empty)
        for o in OBJECTS:
            if o in r_objs and o in c_objs:
                tp[o] += 1
            elif o in c_objs:
                fp[o] += 1
            elif o in r_objs:
                fn[o] += 1
        if r_people == c_people:
            exact += 1
        if abs(r_people - c_people) <= 1:
            within1 += 1

    def f1(t, p, n):
        denom = 2 * t + p + n
        return 1.0 if denom == 0 else 2 * t / denom

    # Overall F1 pools all classes together (micro average).
    # The brief does not say how classes are averaged: this is an assumption.
    micro = f1(sum(tp.values()), sum(fp.values()), sum(fn.values()))
    total = len(keys)
    exact_pct = 100 * exact / total
    within1_pct = 100 * within1 / total

    print(f"seconds compared: {total}   (only in ref: {only_ref}, only in candidate: {only_cand})")
    print(f"error lines: ref={ref_err}, candidate={cand_err}")
    print()
    print("per-class F1 (for diagnosis):")
    for o in OBJECTS:
        print(f"  {o:11s} F1={f1(tp[o], fp[o], fn[o]):.3f}   tp={tp[o]} fp={fp[o]} fn={fn[o]}")
    print()
    print(f"object F1 (micro):      {micro:.4f}   need >= 0.95   {'PASS' if micro >= 0.95 else 'FAIL'}")
    print(f"people exact match:     {exact_pct:.2f}%   need >= 85%     {'PASS' if exact_pct >= 85 else 'FAIL'}")
    print(f"people within +/-1:     {within1_pct:.2f}%   need >= 95%     {'PASS' if within1_pct >= 95 else 'FAIL'}")


if __name__ == "__main__":
    main()