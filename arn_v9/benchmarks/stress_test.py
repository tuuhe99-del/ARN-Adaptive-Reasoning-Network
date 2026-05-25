"""
ARN v9 TIERED STRESS TEST
==========================
Runs the full adversarial stress test against a specific embedding tier.
Usage:
    python3 stress_test_tiered.py nano
    python3 stress_test_tiered.py base
"""

import sys
import os
import time
import shutil
import tempfile
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Parse tier argument
TIER = sys.argv[1] if len(sys.argv) > 1 else 'nano'
print(f"Running stress test with tier: {TIER}")

# Inject tier via env so ARNPlugin picks it up
os.environ['ARN_EMBEDDING_TIER'] = TIER

from arn_v9.plugin import ARNPlugin


class StressResults:
    def __init__(self):
        self.scenarios = {}

    def record(self, scenario, passed, metric, details=""):
        self.scenarios[scenario] = {'passed': passed, 'metric': metric, 'details': details}
        symbol = "✅" if passed else "❌"
        print(f"  {symbol} {scenario}: {metric}")
        if details:
            print(f"      {details}")

    def summary(self):
        passed = sum(1 for s in self.scenarios.values() if s['passed'])
        total = len(self.scenarios)
        print(f"\n{'='*70}")
        print(f"STRESS TEST RESULTS ({TIER}): {passed}/{total} scenarios passed")
        print(f"{'='*70}")
        for name, result in self.scenarios.items():
            symbol = "✅" if result['passed'] else "❌"
            print(f"  {symbol} {name}: {result['metric']}")
        return passed, total


def scenario_cross_session(tmp_dir, results):
    print("\n[1] CROSS-SESSION PERSISTENCE")
    with ARNPlugin(agent_id="persist_test", data_root=tmp_dir, auto_consolidate=False) as p:
        p.store("The user's database password hint is 'blue ocean forty'", importance=0.9)
        p.store("The production server is named 'thunderhead-03'", importance=0.9)
        p.store("Meeting with CFO scheduled for April 22 2026 at 2pm", importance=0.8)

    for session in range(2, 5):
        with ARNPlugin(agent_id="persist_test", data_root=tmp_dir, auto_consolidate=False) as p:
            for i in range(10):
                p.store(f"Session {session} noise {i}: random activity", importance=0.1)

    with ARNPlugin(agent_id="persist_test", data_root=tmp_dir, auto_consolidate=False) as p:
        queries = [
            ("database password hint", "blue ocean forty"),
            ("production server name", "thunderhead-03"),
            ("CFO meeting", "April 22"),
        ]
        found = 0
        for query, expected in queries:
            recalls = p.recall(query, top_k=3)
            if any(expected.lower() in r['content'].lower() for r in recalls):
                found += 1
        results.record("Cross-session persistence", found == len(queries),
                       f"{found}/{len(queries)} facts recalled")


def scenario_distractor(tmp_dir, results):
    print("\n[2] DISTRACTOR RESISTANCE")
    with ARNPlugin(agent_id="distractor_test", data_root=tmp_dir, auto_consolidate=False) as p:
        needles = [
            ("The nuclear launch code is 7x-mango-alpha-91", "nuclear launch code", "mango-alpha"),
            ("Project Stardust has a deadline of June 15 2027", "Project Stardust deadline", "June 15 2027"),
            ("The anomaly was first detected in sector 7G at 03:42 UTC", "anomaly sector detection", "sector 7G"),
            ("Patient Zero's blood type is AB negative Rh-factor", "Patient Zero blood type", "AB negative"),
            ("The decryption key is stored in vault 4429-delta", "decryption key storage", "vault 4429-delta"),
        ]
        import random
        random.seed(42)
        topics = ["weather", "sports", "cooking", "traffic", "stock market"]
        positions = random.sample(range(500), 5)
        idx = 0
        for i in range(500):
            if i in positions and idx < len(needles):
                p.store(needles[idx][0], importance=0.5)
                idx += 1
            else:
                p.store(f"Distractor {i}: {random.choice(topics)} info", importance=0.2)
        
        found = 0
        for _, query, expected in needles:
            recalls = p.recall(query, top_k=5)
            if any(expected.lower() in r['content'].lower() for r in recalls):
                found += 1
        results.record("Distractor resistance", found >= 4,
                       f"{found}/5 needles in 500 haystack")


def scenario_contradiction(tmp_dir, results):
    print("\n[3] CONTRADICTION HANDLING")
    with ARNPlugin(agent_id="contradiction_test", data_root=tmp_dir, auto_consolidate=False) as p:
        contradictions = [
            ("The project budget is $50,000", "50"),
            ("The project budget is $75,000", "75"),
            ("Actually the final project budget is $100,000", "100"),
        ]
        for content, _ in contradictions:
            p.store(content, importance=0.8)
            time.sleep(0.05)
        
        recalls = p.recall("What is the project budget?", top_k=3)
        top_content = recalls[0]['content'] if recalls else ""
        has_latest = "100" in top_content
        results.record("Contradiction handling (most-recent-wins)", has_latest,
                       f"Top={'100' if has_latest else top_content[:30]}")


def scenario_temporal(tmp_dir, results):
    print("\n[4] TEMPORAL REASONING")
    with ARNPlugin(agent_id="temporal_test", data_root=tmp_dir, auto_consolidate=False) as p:
        # Use explicit time_context tags — the agent is expected to tag
        # temporal facts correctly (this is a reasonable API contract)
        p.store("The user used to prefer Python for all development work",
                importance=0.6, time_context='past')
        time.sleep(0.1)
        for i in range(20):
            p.store(f"General activity {i}: routine work", importance=0.2)
        p.store("The user now exclusively uses Rust for new projects",
                importance=0.8, time_context='current')
        
        recalls = p.recall("What programming language does the user currently prefer?", top_k=5)
        rust_rank = python_rank = None
        for i, r in enumerate(recalls):
            if "Rust" in r['content'] and rust_rank is None:
                rust_rank = i
            if "Python" in r['content'] and "used to" in r['content'] and python_rank is None:
                python_rank = i
        
        passed = (rust_rank is not None and (python_rank is None or rust_rank < python_rank))
        results.record("Temporal reasoning (current > past)", passed,
                       f"Rust rank={rust_rank}, Python(past) rank={python_rank}")


def scenario_hallucination(tmp_dir, results):
    print("\n[5] HALLUCINATION REFUSAL")
    with ARNPlugin(agent_id="hallucination_test", data_root=tmp_dir, auto_consolidate=False) as p:
        for f in ["The company logo is blue and gold", "The office has 3 conference rooms",
                  "Coffee is served at 10am daily", "The fiscal year ends in December",
                  "Parking is on the north side of the building"]:
            p.store(f, importance=0.6)
        
        queries = [
            "What is the CEO's favorite ice cream flavor?",
            "What color are the bathroom tiles?",
            "How many employees work on Saturdays?",
        ]
        low_confidence = 0
        top_scores = []
        for q in queries:
            recalls = p.recall(q, top_k=3)
            if recalls:
                top_score = recalls[0]['similarity']
                top_scores.append(top_score)
                # Use model-calibrated confidence tier, not hardcoded threshold
                if recalls[0].get('confidence_tier') == 'low':
                    low_confidence += 1
            else:
                low_confidence += 1
        
        avg = np.mean(top_scores) if top_scores else 0
        results.record("Hallucination refusal", low_confidence >= 2,
                       f"{low_confidence}/3 low-confidence, avg sim={avg:.3f}")


def scenario_paraphrase(tmp_dir, results):
    print("\n[6] PARAPHRASE ROBUSTNESS")
    with ARNPlugin(agent_id="paraphrase_test", data_root=tmp_dir, auto_consolidate=False) as p:
        p.store("Mtr works as a software engineer at a startup called Nebulon", importance=0.8)
        p.store("The server cluster consists of 12 machines running Ubuntu 24.04", importance=0.7)
        p.store("All backups are performed nightly at 2am Eastern time", importance=0.6)
        
        tests = [
            ("Who employs Mtr?", "Nebulon"),
            ("What's Mtr's job?", "engineer"),
            ("How many servers do we have?", "12"),
            ("What OS runs on our infrastructure?", "Ubuntu"),
            ("When do our backups run?", "2am"),
            ("Backup schedule?", "2am"),
        ]
        hits = sum(1 for q, exp in tests
                   if any(exp.lower() in r['content'].lower() for r in p.recall(q, top_k=3)))
        results.record("Paraphrase robustness", hits >= 5,
                       f"{hits}/{len(tests)} hit target")


def scenario_scale(tmp_dir, results):
    print("\n[7] SCALE DEGRADATION")
    scale_results = {}
    for target in [1000, 3000]:
        with ARNPlugin(agent_id=f"scale_{target}", data_root=tmp_dir,
                       auto_consolidate=True, consolidation_threshold=256) as p:
            anchors = [
                ("The secret handshake uses 5 fingers and 2 taps", "secret handshake", "5 fingers"),
                ("Admin access code is phoenix-rising-1847", "admin access code", "phoenix-rising"),
                ("Emergency contact is Dr Yamamoto at extension 4412", "emergency contact", "Yamamoto"),
            ]
            for content, _, _ in anchors:
                p.store(content, importance=0.9)
            
            import random
            random.seed(123)
            templates = ["Meeting notes from {}: discussed project {}",
                         "Email from {} about system {}",
                         "Bug report #{}: issue in module {}"]
            names = ["Alice", "Bob", "Carol", "David"]
            projects = ["alpha", "beta", "gamma", "delta"]
            
            for i in range(target - len(anchors)):
                t = random.choice(templates)
                p.store(t.format(random.choice(names), random.choice(projects)),
                        importance=random.random() * 0.5)
            
            times = []
            hits = 0
            for _, query, expected in anchors:
                start = time.time()
                recalls = p.recall(query, top_k=5)
                times.append(time.time() - start)
                if any(expected.lower() in r['content'].lower() for r in recalls):
                    hits += 1
            
            scale_results[target] = {
                'accuracy': hits / len(anchors),
                'latency_ms': np.mean(times) * 1000,
            }
            print(f"    {target}: acc={scale_results[target]['accuracy']*100:.0f}%, "
                  f"lat={scale_results[target]['latency_ms']:.0f}ms")
    
    passed = all(r['accuracy'] >= 0.8 for r in scale_results.values())
    results.record("Scale degradation (1K & 3K)", passed,
                   f"1K: {scale_results[1000]['accuracy']*100:.0f}% {scale_results[1000]['latency_ms']:.0f}ms | "
                   f"3K: {scale_results[3000]['accuracy']*100:.0f}% {scale_results[3000]['latency_ms']:.0f}ms")


def main():
    print("=" * 70)
    print(f"ARN v9 ADVERSARIAL STRESS TEST — TIER: {TIER}")
    print("=" * 70)
    
    tmp_dir = tempfile.mkdtemp(prefix=f"arn_stress_{TIER}_")
    results = StressResults()
    
    start_total = time.time()
    try:
        scenario_cross_session(tmp_dir, results)
        scenario_distractor(tmp_dir, results)
        scenario_contradiction(tmp_dir, results)
        scenario_temporal(tmp_dir, results)
        scenario_hallucination(tmp_dir, results)
        scenario_paraphrase(tmp_dir, results)
        scenario_scale(tmp_dir, results)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    
    total_time = time.time() - start_total
    passed, total = results.summary()
    print(f"\nTotal test time: {total_time:.1f}s")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
