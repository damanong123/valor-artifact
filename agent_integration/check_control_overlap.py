
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent

MAIN = ROOT / "agent_integration/results/raw/main_runs_main_v1.jsonl"
ATTACK = ROOT / "agent_integration/results/raw/attack_main_runs_attack_main_v1.jsonl"


def load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def text(row):
    return json.dumps(row, ensure_ascii=False).lower()


main = load(MAIN)
attack = load(ATTACK)

# Locate Main paired-control records
paired = [
    r for r in main
    if "paired-control" in text(r)
    or "paired_control" in text(r)
]

# Locate clean control records in the attack experiment
clean = [
    r for r in attack
    if str(r.get("condition", "")).lower() == "clean"
]

print("Main total:", len(main))
print("Attack total:", len(attack))
print("Main paired-control:", len(paired))
print("Attack clean-control:", len(clean))

main_ids = {r.get("run_id") for r in paired}
clean_ids = {r.get("run_id") for r in clean}

overlap = main_ids & clean_ids
print("\nSame run_id count:", len(overlap))

def task_key(r):
    return (
        r.get("task_id"),
        r.get("policy"),
        r.get("repetition", r.get("rep")),
    )

main_keys = {task_key(r) for r in paired}
clean_keys = {task_key(r) for r in clean}

print("Matching task/policy/repetition:", len(main_keys & clean_keys))

print("\nMain paired-control examples:")
for r in paired[:3]:
    print({
        "run_id": r.get("run_id"),
        "task_id": r.get("task_id"),
        "policy": r.get("policy"),
        "rep": r.get("repetition", r.get("rep")),
    })

print("\nAttack clean-control examples:")
for r in clean[:3]:
    print({
        "run_id": r.get("run_id"),
        "task_id": r.get("task_id"),
        "policy": r.get("policy"),
        "rep": r.get("repetition", r.get("rep")),
    })
