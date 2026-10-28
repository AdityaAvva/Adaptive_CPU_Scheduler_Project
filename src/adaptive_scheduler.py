"""
AI/ML-Based Adaptive CPU Scheduling Algorithm for Dynamic Workload Optimization
Educational simulation. Runs on Python 3.10+; charts require matplotlib.
"""
from dataclasses import dataclass, asdict
from collections import deque, defaultdict
import argparse, csv, random, statistics, json, os

WORKLOADS = ("CPU-intensive", "I/O-intensive", "Mixed")
ALGORITHMS = ("FCFS", "SJF", "Priority", "Round Robin")
SCENARIOS = (
    "long-job-convoy",
    "urgent-priority",
    "staggered-response",
    "balanced-mix",
)
GENERATED_SCENARIOS = (
    "baseline",
    "priority-sensitive",
    "response-focused",
    "arrival-mix",
)
CLASSIFIER_SEED = 123

@dataclass
class Process:
    pid: str
    arrival: int
    burst: int
    priority: int
    kind: str = "CPU-intensive"

# These are deliberately small, fixed examples for live demonstration. The
# workload label describes the CPU-burst pattern; no real I/O blocking occurs.
HAND_DESIGNED_SCENARIOS = {
    "long-job-convoy": {
        "workload": "CPU-intensive",
        "description": "A long job and short jobs are ready together, exposing convoy effects.",
        "processes": (
            Process("P1", 0, 18, 3, "CPU-intensive"),
            Process("P2", 0, 2, 3, "CPU-intensive"),
            Process("P3", 0, 3, 3, "CPU-intensive"),
            Process("P4", 1, 1, 3, "CPU-intensive"),
            Process("P5", 2, 2, 3, "CPU-intensive"),
        ),
    },
    "urgent-priority": {
        "workload": "Mixed",
        "description": "Urgent processes use the smallest numeric priority and may arrive later.",
        "processes": (
            Process("P1", 0, 10, 5, "Mixed"),
            Process("P2", 0, 4, 4, "Mixed"),
            Process("P3", 0, 6, 1, "Mixed"),
            Process("P4", 1, 2, 1, "Mixed"),
            Process("P5", 3, 3, 2, "Mixed"),
        ),
    },
    "staggered-response": {
        "workload": "Mixed",
        "description": "Staggered arrivals make first response and fairness visible under Round Robin.",
        "processes": (
            Process("P1", 0, 8, 3, "Mixed"),
            Process("P2", 1, 6, 3, "Mixed"),
            Process("P3", 2, 4, 3, "Mixed"),
            Process("P4", 3, 2, 3, "Mixed"),
            Process("P5", 4, 3, 3, "Mixed"),
        ),
    },
    "balanced-mix": {
        "workload": "I/O-intensive",
        "description": "Short CPU bursts with varied arrivals and priorities compare all policies fairly.",
        "processes": (
            Process("P1", 0, 5, 5, "I/O-intensive"),
            Process("P2", 0, 1, 1, "I/O-intensive"),
            Process("P3", 1, 4, 3, "I/O-intensive"),
            Process("P4", 2, 2, 2, "I/O-intensive"),
            Process("P5", 4, 3, 4, "I/O-intensive"),
        ),
    },
}

def generate_workload(kind, n=10, seed=42):
    if kind not in WORKLOADS:
        raise ValueError(f"Unknown workload kind: {kind}")
    if not isinstance(n, int) or isinstance(n, bool) or n < 1:
        raise ValueError("The number of processes must be a positive integer.")
    rng = random.Random(seed)
    ps = []
    for i in range(n):
        arrival = rng.randint(0, max(1, n // 3))
        if kind == "CPU-intensive":
            burst = rng.randint(8, 20)
        elif kind == "I/O-intensive":
            burst = rng.randint(1, 5)
        else:
            burst = rng.randint(2, 16)
        ps.append(Process(f"P{i+1}", arrival, burst, rng.randint(1, 5), kind))
    return sorted(ps, key=lambda p: (p.arrival, p.pid))

def generate_scenario(scenario, kind, n=10, seed=42):
    """Generate a reproducible scenario-specific process set.

    The baseline deliberately delegates to the original generator so its
    existing values remain unchanged. Other scenarios vary arrivals, bursts,
    and priorities while retaining the same Process model and workload labels.
    """
    if scenario not in GENERATED_SCENARIOS:
        raise ValueError(f"Unknown benchmark scenario: {scenario}")
    if scenario == "baseline":
        return generate_workload(kind, n, seed)
    if kind not in WORKLOADS:
        raise ValueError(f"Unknown workload kind: {kind}")
    if not isinstance(n, int) or isinstance(n, bool) or n < 1:
        raise ValueError("The number of processes must be a positive integer.")

    rng = random.Random((seed * 1009) + (GENERATED_SCENARIOS.index(scenario) * 9176))
    processes = []
    for index in range(n):
        if kind == "CPU-intensive":
            burst_range = (8, 20)
        elif kind == "I/O-intensive":
            burst_range = (1, 5)
        else:
            burst_range = (2, 16)
        if scenario == "priority-sensitive":
            arrival = rng.randint(0, max(1, n // 2))
            burst = rng.randint(*burst_range)
            # Priority values are intentionally varied so urgent arrivals are
            # observable in the same process set as the other algorithms.
            priority = rng.choices((1, 2, 3, 4, 5), weights=(4, 3, 2, 1, 1))[0]
        elif scenario == "response-focused":
            arrival = rng.randint(0, max(1, n // 3))
            burst = rng.randint(*burst_range)
            priority = rng.randint(1, 5)
        else:  # arrival-mix
            arrival = rng.randint(0, n)
            burst = rng.randint(*burst_range)
            priority = rng.randint(1, 5)
        processes.append(Process(f"P{index + 1}", arrival, burst, priority, kind))
    return sorted(processes, key=lambda p: (p.arrival, p.pid))

def get_hand_designed_scenario(scenario):
    """Return a copy of one documented, deterministic demonstration input."""
    if scenario not in HAND_DESIGNED_SCENARIOS:
        raise ValueError(f"Unknown hand-designed scenario: {scenario}")
    return [Process(**asdict(process)) for process in HAND_DESIGNED_SCENARIOS[scenario]["processes"]]

def _feature_vector(bursts, arrivals):
    n = len(bursts)
    if n == 0:
        raise ValueError("Cannot extract workload features from an empty workload.")
    mean_burst = sum(bursts) / n
    variance = sum((burst - mean_burst) ** 2 for burst in bursts) / n
    short_burst_ratio = sum(burst <= 5 for burst in bursts) / n
    arrival_span = (max(arrivals) - min(arrivals)) / n if n > 1 else 0
    return [mean_burst, variance ** 0.5, short_burst_ratio, arrival_span]

def features(processes):
    return _feature_vector(
        [p.burst for p in processes],
        [p.arrival for p in processes],
    )

class WorkloadClassifier:
    """Small nearest-centroid classifier trained on generated labelled examples."""
    def __init__(self):
        self.centroids = {}
        self.scales = [1.0] * 4

    def fit(self, examples):
        grouped = defaultdict(list)
        for x, label in examples:
            grouped[label].append(x)
        if not grouped:
            raise ValueError("Classifier training requires at least one example.")
        self.centroids = {label: [sum(row[j] for row in rows)/len(rows) for j in range(4)]
                          for label, rows in grouped.items()}
        self.scales = []
        for j in range(4):
            vals = [x[j] for x, _ in examples]
            self.scales.append(max(max(vals)-min(vals), 1.0))
        return self

    def predict_one(self, x):
        if not self.centroids:
            raise ValueError("Classifier must be trained before prediction.")
        def distance(c):
            return sum(((x[j] - c[j]) / self.scales[j]) ** 2 for j in range(4))
        return min(self.centroids, key=lambda label: distance(self.centroids[label]))

    def predict(self, X):
        return [self.predict_one(x) for x in X]

def train_classifier(seed=123, samples_per_class=120):
    if not isinstance(samples_per_class, int) or isinstance(samples_per_class, bool) or samples_per_class < 1:
        raise ValueError("Samples per class must be a positive integer.")
    rng = random.Random(seed)
    examples = []
    for label in WORKLOADS:
        for _ in range(samples_per_class):
            n = rng.randint(6, 20)
            if label == "CPU-intensive":
                bursts = [rng.randint(8, 20) for _ in range(n)]
            elif label == "I/O-intensive":
                bursts = [rng.randint(1, 5) for _ in range(n)]
            else:
                bursts = [rng.randint(2, 16) for _ in range(n)]
            arrivals = [rng.randint(0, max(1, n//3)) for _ in range(n)]
            examples.append((_feature_vector(bursts, arrivals), label))
    rng.shuffle(examples)
    return WorkloadClassifier().fit(examples), examples

def run_schedule(processes, algorithm, quantum=4):
    """Discrete-time CPU scheduler; process burst is CPU service demand.
    Returns per-process metrics and timeline. SJF is non-preemptive.
    """
    if algorithm not in ALGORITHMS:
        raise ValueError("Unknown algorithm: " + algorithm)
    if algorithm == "Round Robin" and (
        not isinstance(quantum, int) or isinstance(quantum, bool) or quantum < 1
    ):
        raise ValueError("Round Robin quantum must be a positive integer.")
    if len({p.pid for p in processes}) != len(processes):
        raise ValueError("Process IDs must be unique.")
    if any(p.arrival < 0 or p.burst < 1 for p in processes):
        raise ValueError("Process arrivals must be non-negative and bursts must be positive.")
    ps = [Process(**asdict(p)) for p in processes]
    remaining = {p.pid: p.burst for p in ps}
    first_start, completion = {}, {}
    timeline = []
    t = 0
    ready = []
    arrived = set()
    by_pid = {p.pid:p for p in ps}
    if algorithm == "FCFS":
        order = sorted(ps, key=lambda p:(p.arrival, p.pid))
        for p in order:
            t = max(t, p.arrival)
            first_start[p.pid] = t
            timeline.append((t, t+p.burst, p.pid))
            t += p.burst
            completion[p.pid] = t
    elif algorithm in ("SJF", "Priority"):
        pending = set(p.pid for p in ps)
        while pending:
            available = [by_pid[x] for x in pending if by_pid[x].arrival <= t]
            if not available:
                t = min(by_pid[x].arrival for x in pending)
                available = [by_pid[x] for x in pending if by_pid[x].arrival <= t]
            if algorithm == "SJF":
                p = min(available, key=lambda p:(p.burst, p.arrival, p.pid))
            else:
                p = min(available, key=lambda p:(p.priority, p.arrival, p.pid))
            first_start[p.pid] = t
            timeline.append((t, t+p.burst, p.pid))
            t += p.burst
            completion[p.pid] = t
            pending.remove(p.pid)
    elif algorithm == "Round Robin":
        pending = sorted(ps, key=lambda p:(p.arrival,p.pid))
        q = deque()
        idx = 0
        while idx < len(pending) or q:
            if not q and idx < len(pending) and t < pending[idx].arrival:
                t = pending[idx].arrival
            while idx < len(pending) and pending[idx].arrival <= t:
                q.append(pending[idx]); idx += 1
            if not q:
                continue
            p = q.popleft()
            if p.pid not in first_start:
                first_start[p.pid] = t
            duration = min(quantum, remaining[p.pid])
            timeline.append((t, t+duration, p.pid))
            t += duration
            remaining[p.pid] -= duration
            while idx < len(pending) and pending[idx].arrival <= t:
                q.append(pending[idx]); idx += 1
            if remaining[p.pid] > 0:
                q.append(p)
            else:
                completion[p.pid] = t
    metrics = []
    for p in ps:
        finish = completion[p.pid]
        turnaround = finish - p.arrival
        waiting = turnaround - p.burst
        response = first_start[p.pid] - p.arrival
        metrics.append({"pid":p.pid, "arrival":p.arrival, "burst":p.burst,
                        "priority":p.priority, "completion":finish,
                        "waiting":waiting, "turnaround":turnaround, "response":response})
    makespan = max((end for _, end, _ in timeline), default=0)
    total_cpu = sum(p.burst for p in ps)
    simulation_start = min((p.arrival for p in ps), default=0)
    simulation_duration = makespan - simulation_start
    summary = {
        "algorithm": algorithm,
        "avg_waiting": statistics.mean(m["waiting"] for m in metrics) if metrics else 0,
        "avg_turnaround": statistics.mean(m["turnaround"] for m in metrics) if metrics else 0,
        "avg_response": statistics.mean(m["response"] for m in metrics) if metrics else 0,
        "throughput": len(ps)/simulation_duration if simulation_duration else 0,
        "cpu_utilization_pct": 100*total_cpu/simulation_duration if simulation_duration else 0,
        "makespan": makespan,
        "simulation_duration": simulation_duration,
    }
    highest_priority = min((p.priority for p in ps), default=None)
    urgent = [m for m in metrics if m["priority"] == highest_priority]
    summary["highest_priority_value"] = highest_priority
    summary["highest_priority_processes"] = "; ".join(m["pid"] for m in urgent)
    summary["avg_highest_priority_response"] = (
        statistics.mean(m["response"] for m in urgent) if urgent else 0
    )
    return metrics, timeline, summary

def run_demo(seed=42, n=10, quantum=4):
    if not isinstance(quantum, int) or isinstance(quantum, bool) or quantum < 1:
        raise ValueError("Round Robin quantum must be a positive integer.")
    classifier, _ = train_classifier(seed=CLASSIFIER_SEED)
    results = []
    for kind in WORKLOADS:
        processes = generate_workload(kind, n=n, seed=seed)
        predicted = classifier.predict_one(features(processes))
        # Explicit, explainable mapping from predicted workload to policy.
        policy = {"CPU-intensive":"SJF", "I/O-intensive":"Round Robin", "Mixed":"Priority"}[predicted]
        for alg in ALGORITHMS:
            _, _, summary = run_schedule(processes, alg, quantum)
            summary.update({"actual_workload":kind, "predicted_workload":predicted,
                            "adaptive_policy":policy, "is_adaptive":alg==policy,
                            "seed":seed, "classifier_seed":CLASSIFIER_SEED,
                            "process_count":n, "quantum":quantum})
            results.append(summary)
    return results

def _winner_names(rows, metric):
    best = min(row[metric] for row in rows)
    return tuple(row["algorithm"] for row in rows if row[metric] == best)

def summarize_winners(rows):
    """Return measured metric winners grouped by scenario and workload."""
    summaries = []
    groups = {}
    for row in rows:
        groups.setdefault((row["scenario"], row["actual_workload"]), []).append(row)
    for (scenario, workload), group in sorted(groups.items()):
        summaries.append({
            "scenario": scenario,
            "actual_workload": workload,
            "avg_waiting_winners": _winner_names(group, "avg_waiting"),
            "avg_turnaround_winners": _winner_names(group, "avg_turnaround"),
            "avg_response_winners": _winner_names(group, "avg_response"),
            "priority_response_winners": _winner_names(
                group, "avg_highest_priority_response"
            ),
            "throughput_winners": tuple(
                row["algorithm"] for row in group
                if row["throughput"] == max(item["throughput"] for item in group)
            ),
            "cpu_utilization_winners": tuple(
                row["algorithm"] for row in group
                if row["cpu_utilization_pct"] == max(item["cpu_utilization_pct"] for item in group)
            ),
        })
    return summaries

def run_benchmark(seed=42, n=5, quantum=4):
    """Run every algorithm on identical hand-designed inputs."""
    if not isinstance(quantum, int) or isinstance(quantum, bool) or quantum < 1:
        raise ValueError("Round Robin quantum must be a positive integer.")
    if n != 5:
        raise ValueError("Hand-designed benchmark scenarios contain exactly five processes.")
    classifier, _ = train_classifier(seed=CLASSIFIER_SEED)
    results = []
    for scenario in SCENARIOS:
        definition = HAND_DESIGNED_SCENARIOS[scenario]
        kind = definition["workload"]
        processes = get_hand_designed_scenario(scenario)
        predicted = classifier.predict_one(features(processes))
        policy = {
            "CPU-intensive": "SJF",
            "I/O-intensive": "Round Robin",
            "Mixed": "Priority",
        }[predicted]
        process_inputs = json.dumps([asdict(process) for process in processes], separators=(",", ":"))
        for algorithm in ALGORITHMS:
            _, _, summary = run_schedule(processes, algorithm, quantum)
            summary.update({
                "scenario": scenario,
                "scenario_description": definition["description"],
                "actual_workload": kind,
                "predicted_workload": predicted,
                "adaptive_policy": policy,
                "is_adaptive": algorithm == policy,
                "seed": seed,
                "classifier_seed": CLASSIFIER_SEED,
                "process_count": len(processes),
                "quantum": quantum,
                "process_inputs": process_inputs,
            })
            results.append(summary)
    winners = summarize_winners(results)
    winner_lookup = {
        (item["scenario"], item["actual_workload"]): item
        for item in winners
    }
    for row in results:
        winner = winner_lookup[(row["scenario"], row["actual_workload"])]
        for key in (
            "avg_waiting_winners",
            "avg_turnaround_winners",
            "avg_response_winners",
            "priority_response_winners",
            "throughput_winners",
            "cpu_utilization_winners",
        ):
            row[key] = "; ".join(winner[key])
    return results, winners

def save_results(rows, outdir="results"):
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "benchmark_results.csv")
    fields = list(rows[0].keys()) if rows else []
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    with open(os.path.join(outdir, "benchmark_results.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    return path

def save_visualizations(rows, outdir="results"):
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "Matplotlib is required to generate charts. Install it with "
            "'py -m pip install matplotlib'."
        ) from exc

    scenarios = {row.get("scenario", SCENARIOS[0]) for row in rows}
    scenario = SCENARIOS[0] if SCENARIOS[0] in scenarios else next(iter(scenarios))
    chart_rows = [row for row in rows if row.get("scenario", "baseline") == scenario]
    workloads = sorted({row["actual_workload"] for row in chart_rows})
    algorithms = ALGORITHMS
    charts = (
        ("avg_waiting", "Average waiting time", "avg_waiting_comparison.png"),
        ("avg_turnaround", "Average turnaround time", "avg_turnaround_comparison.png"),
    )
    x_positions = list(range(len(workloads)))
    width = 0.8 / len(algorithms)
    for metric, title, filename in charts:
        figure, axis = plt.subplots(figsize=(10, 5))
        for index, algorithm in enumerate(algorithms):
            values = []
            for workload in workloads:
                row = next(
                    item for item in chart_rows
                    if item["actual_workload"] == workload and item["algorithm"] == algorithm
                )
                values.append(row[metric])
            offsets = [x + (index - (len(algorithms) - 1) / 2) * width for x in x_positions]
            axis.bar(offsets, values, width, label=algorithm)
        axis.set_title(title)
        axis.set_ylabel("Time units")
        axis.set_xticks(x_positions, workloads)
        axis.legend()
        figure.tight_layout()
        figure.savefig(os.path.join(outdir, filename), dpi=150)
        plt.close(figure)
    return [os.path.join(outdir, filename) for _, _, filename in charts]

def print_results(rows):
    print("\nAI/ML-BASED ADAPTIVE CPU SCHEDULER")
    print("="*74)
    seen = set()
    for r in rows:
        key = (r.get("scenario", "baseline"), r["actual_workload"], r["algorithm"])
        if key in seen: continue
        seen.add(key)
        print(f'{r.get("scenario", "baseline"):20} | {r["actual_workload"]:16} | '
              f'{r["algorithm"]:12} | '
              f'Wait {r["avg_waiting"]:6.2f} | Turnaround {r["avg_turnaround"]:6.2f} | '
              f'Response {r["avg_response"]:6.2f} | Throughput {r["throughput"]:6.3f} | '
              f'CPU {r["cpu_utilization_pct"]:6.1f}%')
    for kind in WORKLOADS:
        r = next(x for x in rows if x["actual_workload"] == kind)
        print(f'Adaptive choice for {kind}: predicted={r["predicted_workload"]}, policy={r["adaptive_policy"]}')
    print("\nCSV/JSON results saved in results/")
    print("Comparison charts saved in results/")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--processes", type=int, default=5)
    parser.add_argument("--quantum", type=int, default=4)
    args = parser.parse_args()
    if args.processes != 5:
        parser.error("hand-designed benchmark scenarios require --processes 5")
    if args.quantum < 1:
        parser.error("--quantum must be at least 1")
    rows, winners = run_benchmark(args.seed, args.processes, args.quantum)
    print_results(rows)
    print("\nMeasured metric winners:")
    for winner in winners:
        print(
            f'{winner["scenario"]:20} | {winner["actual_workload"]:16} | '
            f'Wait {", ".join(winner["avg_waiting_winners"])} | '
            f'Turnaround {", ".join(winner["avg_turnaround_winners"])} | '
            f'Response {", ".join(winner["avg_response_winners"])} | '
            f'Priority response {", ".join(winner["priority_response_winners"])} | '
            f'Throughput {", ".join(winner["throughput_winners"])} | '
            f'CPU {", ".join(winner["cpu_utilization_winners"])}'
        )
    save_results(rows)
    save_visualizations(rows)

if __name__ == "__main__":
    main()
