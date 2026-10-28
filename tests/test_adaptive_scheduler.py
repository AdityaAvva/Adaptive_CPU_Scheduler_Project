import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import adaptive_scheduler as scheduler


class SchedulerTests(unittest.TestCase):
    def test_fcfs_respects_arrival_order_and_cpu_idle_periods(self):
        processes = [
            scheduler.Process("P1", 3, 2, 1),
            scheduler.Process("P2", 7, 1, 1),
            scheduler.Process("P3", 3, 1, 1),
        ]
        metrics, timeline, summary = scheduler.run_schedule(processes, "FCFS")
        self.assertEqual(timeline, [(3, 5, "P1"), (5, 6, "P3"), (7, 8, "P2")])
        self.assertEqual(
            {row["pid"]: row["completion"] for row in metrics},
            {"P1": 5, "P2": 8, "P3": 6},
        )
        self.assertEqual(summary["simulation_duration"], 5)
        self.assertEqual(summary["cpu_utilization_pct"], 80)

    def test_sjf_and_priority_choose_best_available_process(self):
        processes = [
            scheduler.Process("P1", 0, 5, 5),
            scheduler.Process("P2", 1, 3, 1),
            scheduler.Process("P3", 2, 1, 3),
        ]
        _, sjf_timeline, _ = scheduler.run_schedule(processes, "SJF")
        _, priority_timeline, _ = scheduler.run_schedule(processes, "Priority")
        self.assertEqual(
            sjf_timeline,
            [(0, 5, "P1"), (5, 6, "P3"), (6, 9, "P2")],
        )
        self.assertEqual(
            priority_timeline,
            [(0, 5, "P1"), (5, 8, "P2"), (8, 9, "P3")],
        )

    def test_round_robin_quantum_arrival_and_idle_handling(self):
        processes = [
            scheduler.Process("P1", 0, 5, 1),
            scheduler.Process("P2", 1, 3, 1),
            scheduler.Process("P3", 10, 1, 1),
        ]
        metrics, timeline, summary = scheduler.run_schedule(
            processes, "Round Robin", quantum=2
        )
        self.assertEqual(
            timeline,
            [
                (0, 2, "P1"),
                (2, 4, "P2"),
                (4, 6, "P1"),
                (6, 7, "P2"),
                (7, 8, "P1"),
                (10, 11, "P3"),
            ],
        )
        self.assertEqual(
            {row["pid"]: row["completion"] for row in metrics},
            {"P1": 8, "P2": 7, "P3": 11},
        )
        self.assertEqual(summary["simulation_duration"], 11)

    def test_schedule_metrics_match_timeline_and_process_work(self):
        processes = [
            scheduler.Process("P1", 0, 5, 1),
            scheduler.Process("P2", 1, 3, 1),
            scheduler.Process("P3", 10, 1, 1),
        ]
        for algorithm in scheduler.ALGORITHMS:
            with self.subTest(algorithm=algorithm):
                metrics, timeline, summary = scheduler.run_schedule(
                    processes, algorithm, quantum=2
                )
                self.assertEqual(len(metrics), len(processes))
                self.assertEqual({row["pid"] for row in metrics}, {p.pid for p in processes})
                self.assertEqual(
                    {pid: sum(end - start for start, end, owner in timeline if owner == pid)
                     for pid in (p.pid for p in processes)},
                    {p.pid: p.burst for p in processes},
                )
                for row in metrics:
                    self.assertEqual(
                        row["turnaround"],
                        row["completion"] - row["arrival"],
                    )
                    self.assertEqual(
                        row["waiting"],
                        row["turnaround"] - row["burst"],
                    )
                    first_start = min(start for start, _, pid in timeline if pid == row["pid"])
                    self.assertEqual(row["response"], first_start - row["arrival"])
                for previous, current in zip(timeline, timeline[1:]):
                    self.assertLessEqual(previous[1], current[0])
                self.assertLessEqual(summary["cpu_utilization_pct"], 100)

    def test_round_robin_rejects_non_positive_quantum(self):
        processes = [scheduler.Process("P1", 0, 2, 1)]
        for quantum in (0, -1):
            with self.subTest(quantum=quantum):
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    scheduler.run_schedule(processes, "Round Robin", quantum)

    def test_utilization_and_throughput_use_first_arrival_as_start(self):
        processes = [scheduler.Process("P1", 5, 2, 1)]
        _, _, summary = scheduler.run_schedule(processes, "FCFS")
        self.assertEqual(summary["makespan"], 7)
        self.assertEqual(summary["simulation_duration"], 2)
        self.assertEqual(summary["cpu_utilization_pct"], 100)
        self.assertEqual(summary["throughput"], 0.5)

    def test_benchmarks_have_one_adaptive_marker_per_workload(self):
        rows = scheduler.run_demo(seed=42, n=6, quantum=3)
        self.assertEqual(len(rows), len(scheduler.WORKLOADS) * len(scheduler.ALGORITHMS))
        for workload in scheduler.WORKLOADS:
            workload_rows = [row for row in rows if row["actual_workload"] == workload]
            self.assertEqual(
                sum(row["is_adaptive"] for row in workload_rows),
                1,
            )
            adaptive_row = next(row for row in workload_rows if row["is_adaptive"])
            self.assertEqual(adaptive_row["algorithm"], adaptive_row["adaptive_policy"])

    def test_classifier_selects_labels_for_representative_generated_workloads(self):
        classifier, _ = scheduler.train_classifier()
        for workload in scheduler.WORKLOADS:
            with self.subTest(workload=workload):
                processes = scheduler.generate_workload(workload, n=6, seed=42)
                self.assertEqual(
                    classifier.predict_one(scheduler.features(processes)),
                    workload,
                )

    def test_benchmark_output_and_visualizations_are_written(self):
        rows = scheduler.run_demo(seed=9, n=4, quantum=2)
        with tempfile.TemporaryDirectory() as output_dir:
            csv_path = scheduler.save_results(rows, output_dir)
            chart_paths = scheduler.save_visualizations(rows, output_dir)

            with open(csv_path, newline="", encoding="utf-8") as results_file:
                exported_rows = list(csv.DictReader(results_file))
            self.assertEqual(len(exported_rows), len(rows))
            self.assertTrue(all(row["seed"] == "9" for row in exported_rows))

            json_path = Path(output_dir) / "benchmark_results.json"
            with json_path.open(encoding="utf-8") as results_file:
                self.assertEqual(json.load(results_file), rows)
            self.assertEqual(len(chart_paths), 2)
            self.assertTrue(all(Path(path).is_file() for path in chart_paths))

    def test_benchmark_is_reproducible_for_same_configuration(self):
        first = scheduler.run_demo(seed=17, n=5, quantum=2)
        second = scheduler.run_demo(seed=17, n=5, quantum=2)
        self.assertEqual(first, second)

    def test_scenarios_are_reproducible_and_change_process_inputs(self):
        baseline = scheduler.generate_scenario("baseline", "Mixed", n=8, seed=42)
        priority = scheduler.generate_scenario(
            "priority-sensitive", "Mixed", n=8, seed=42
        )
        self.assertEqual(
            priority,
            scheduler.generate_scenario("priority-sensitive", "Mixed", n=8, seed=42),
        )
        self.assertNotEqual(baseline, priority)
        self.assertEqual({p.pid for p in priority}, {f"P{i}" for i in range(1, 9)})
        self.assertEqual(
            priority,
            sorted(priority, key=lambda p: (p.arrival, p.pid)),
        )
        self.assertNotEqual(
            priority,
            scheduler.generate_scenario("priority-sensitive", "I/O-intensive", n=8, seed=42),
        )

    def test_benchmark_runs_all_algorithms_on_each_scenario_input(self):
        rows, winners = scheduler.run_benchmark(seed=17, n=5, quantum=2)
        expected_rows = (
            len(scheduler.SCENARIOS) * len(scheduler.ALGORITHMS)
        )
        self.assertEqual(len(rows), expected_rows)
        self.assertEqual(len(winners), len(scheduler.SCENARIOS))
        for scenario in scheduler.SCENARIOS:
            group = [row for row in rows if row["scenario"] == scenario]
            self.assertEqual({row["algorithm"] for row in group}, set(scheduler.ALGORITHMS))
            self.assertEqual(len({row["process_inputs"] for row in group}), 1)
            self.assertEqual(len({row["seed"] for row in group}), 1)
            self.assertTrue(all(row["scenario"] == scenario for row in group))

    def test_hand_designed_scenarios_have_documented_inputs(self):
        self.assertEqual(set(scheduler.HAND_DESIGNED_SCENARIOS), set(scheduler.SCENARIOS))
        self.assertEqual(
            [(p.pid, p.arrival, p.burst, p.priority)
             for p in scheduler.get_hand_designed_scenario("long-job-convoy")],
            [("P1", 0, 18, 3), ("P2", 0, 2, 3), ("P3", 0, 3, 3),
             ("P4", 1, 1, 3), ("P5", 2, 2, 3)],
        )
        urgent = scheduler.get_hand_designed_scenario("urgent-priority")
        self.assertEqual(min(p.priority for p in urgent), 1)
        self.assertEqual(
            [p.pid for p in urgent if p.priority == 1],
            ["P3", "P4"],
        )

    def test_priority_response_metric_includes_all_minimum_priority_ties(self):
        processes = scheduler.get_hand_designed_scenario("urgent-priority")
        metrics, _, summary = scheduler.run_schedule(processes, "Priority", quantum=2)
        urgent = [row for row in metrics if row["priority"] == 1]
        self.assertEqual(summary["highest_priority_processes"], "P3; P4")
        self.assertEqual(
            summary["avg_highest_priority_response"],
            sum(row["response"] for row in urgent) / len(urgent),
        )

    def test_winners_are_measured_and_ties_are_preserved(self):
        rows, winners = scheduler.run_benchmark(seed=17, n=5, quantum=2)
        for winner in winners:
            group = [
                row for row in rows
                if row["scenario"] == winner["scenario"]
                and row["actual_workload"] == winner["actual_workload"]
            ]
            for metric, direction, key in (
                ("avg_waiting", min, "avg_waiting_winners"),
                ("avg_turnaround", min, "avg_turnaround_winners"),
                ("avg_response", min, "avg_response_winners"),
                ("avg_highest_priority_response", min, "priority_response_winners"),
            ):
                measured = direction(row[metric] for row in group)
                expected = tuple(
                    row["algorithm"] for row in group if row[metric] == measured
                )
                self.assertEqual(winner[key], expected)
            self.assertTrue(winner["avg_waiting_winners"])

    def test_scenario_benchmark_csv_contains_configuration_and_winners(self):
        rows, _ = scheduler.run_benchmark(seed=9, n=5, quantum=2)
        with tempfile.TemporaryDirectory() as output_dir:
            csv_path = scheduler.save_results(rows, output_dir)
            with open(csv_path, newline="", encoding="utf-8") as results_file:
                exported = list(csv.DictReader(results_file))
        self.assertEqual(len(exported), len(rows))
        required = {
            "scenario",
            "seed",
            "process_count",
            "quantum",
            "avg_waiting_winners",
            "avg_turnaround_winners",
            "avg_response_winners",
            "priority_response_winners",
            "throughput_winners",
            "cpu_utilization_winners",
        }
        self.assertTrue(required.issubset(exported[0]))
        self.assertEqual(
            {row["scenario"] for row in exported},
            set(scheduler.SCENARIOS),
        )
        self.assertTrue(all(row["avg_waiting_winners"] for row in exported))


if __name__ == "__main__":
    unittest.main()
