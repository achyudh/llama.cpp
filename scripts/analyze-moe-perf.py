#!/usr/bin/env python3
"""Summarize numeric MoE telemetry without displaying model text."""

import argparse
import collections
import json
import re
from pathlib import Path


FIELDS = set("schema us task slot phase role layer type total hits misses evictions hit_bytes fill_bytes fill_calls bypass_bytes bypass_calls bypass_large bypass_other padding_bytes d2d_bytes d2d_calls id_bytes id_calls id_wait_us plan_us submit_us copy_le64k copy_le1m copy_le4m copy_gt4m width graphs".split())
REQUEST_FIELDS = set("schema us task slot final reason state stop input context cached processed generated prompt_ms decode_ms drafted accepted verify_steps accepted_pos tested_pos checkpoint_restores replay_verifications target_wall_us target_calls verify_widths".split())
HOST_REGISTRATION_FIELDS = set("schema budget registered_bytes attempted completed failed skipped us".split())
LABELS = set("schema us task slot phase role layer type total".split())
PAIR = re.compile(r"([a-z0-9_]+)=(-?\d+)")


def numeric(value):
    if isinstance(value, (int, float, bool)):
        return True
    if isinstance(value, list):
        return all(numeric(item) for item in value)
    return False


def summarize(path):
    counters = collections.defaultdict(collections.Counter)
    widths = collections.defaultdict(collections.Counter)
    requests = {}
    mtp = {}
    current = None
    invalid = 0
    registrations = []
    with path.open(errors="replace") as stream:
        for line in stream:
            match = re.search(r"moe_host_registration: ((?:[a-z0-9_]+=-?\d+ ?)+)\s*$", line)
            if match:
                record = {k: int(v) for k, v in PAIR.findall(match[1])}
                if record.keys() != HOST_REGISTRATION_FIELDS or record.get("schema") != 1 or any(v < 0 for v in record.values()):
                    invalid += 1
                else:
                    registrations.append(record)
                continue
            match = re.search(r"request_perf: (\{.*\})\s*$", line)
            if match:
                try:
                    record = json.loads(match[1])
                    if not isinstance(record, dict) or not record.keys() <= REQUEST_FIELDS or not all(numeric(v) for v in record.values()) or record.get("schema") != 1:
                        invalid += 1
                        continue
                    current = (record["task"], record["slot"])
                    requests[current] = record
                except (ValueError, KeyError):
                    invalid += 1
                continue
            match = re.search(r"(moe_perf|moe_width): ((?:[a-z0-9_]+=-?\d+ ?)+)\s*$", line)
            if match:
                record = {k: int(v) for k, v in PAIR.findall(match[2])}
                if not record.keys() <= FIELDS or record.get("schema") != 1:
                    invalid += 1
                    continue
                try:
                    key = (record["task"], record["slot"], record["phase"], record["role"])
                    if match[1] == "moe_width":
                        widths[key][record["width"]] += record["graphs"]
                    elif record["total"] == 0:
                        counters[key].update({k: v for k, v in record.items() if k not in LABELS})
                except KeyError:
                    invalid += 1
                continue
            match = re.search(r"dur\(b,g,a\) = ([\d.]+), ([\d.]+), ([\d.]+) ms", line)
            if match and current is not None:
                value = [float(v) for v in match.groups()]
                first, _ = mtp.get(current, (value, value))
                mtp[current] = (first, value)
    result = {"schema": 1, "invalid_records": invalid, "host_registrations": registrations, "requests": [], "phases": []}
    for key, record in requests.items():
        item = dict(record)
        if key in mtp:
            first, last = mtp[key]
            item["mtp_begin_generate_accept_ms"] = [b - a for a, b in zip(first, last)]
        result["requests"].append(item)
    for key in sorted(counters.keys() | widths.keys()):
        task, slot, phase, role = key
        item = dict(task=task, slot=slot, phase=phase, role=role, **counters[key])
        item["width_graphs"] = sorted(widths[key].items())
        hits, misses = item.get("hits", 0), item.get("misses", 0)
        if hits + misses:
            item["hit_fraction"] = hits / (hits + misses)
        hit_bytes, fill_bytes = item.get("hit_bytes", 0), item.get("fill_bytes", 0)
        if hit_bytes + fill_bytes:
            item["byte_hit_fraction"] = hit_bytes / (hit_bytes + fill_bytes)
        result["phases"].append(item)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    args = parser.parse_args()
    path = args.log / "server.log" if args.log.is_dir() else args.log
    print(json.dumps(summarize(path), indent=2))


if __name__ == "__main__":
    main()
