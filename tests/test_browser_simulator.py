import unittest
from pathlib import Path


class BrowserSimulatorContractTests(unittest.TestCase):
    """Regression checks for the standalone browser simulator contract.

    Runtime scheduling behavior is exercised in the local browser smoke test;
    these checks prevent the multi-burst model from being accidentally replaced
    by the former single-burst implementation.
    """

    @classmethod
    def setUpClass(cls):
        cls.html = (
            Path(__file__).resolve().parents[1] / "website" / "index.html"
        ).read_text(encoding="utf-8")

    def test_workload_generator_defines_real_io_sequences(self):
        self.assertIn("cpuBursts", self.html)
        self.assertIn("ioWaits", self.html)
        self.assertIn("burstCount = 3", self.html)
        self.assertIn("10 + Math.floor(random() * 6)", self.html)
        self.assertIn("burstIndex === 0 ? 1", self.html)
        self.assertIn("ioWaitRatio", self.html)
        self.assertIn("8 + Math.floor(random() * 13)", self.html)
        self.assertIn("burstIndex % 2 === 0", self.html)

    def test_workload_generation_keeps_categories_distinct(self):
        self.assertIn('if (kind === "CPU-intensive")', self.html)
        self.assertIn('else if (kind === "I/O-intensive")', self.html)
        self.assertIn('kind === "I/O-intensive"\n          ? Math.floor(random() * 21)', self.html)
        self.assertIn('const burstCount = 2 + Math.floor(random() * 2)', self.html)
        self.assertIn('const burstCount = 3 + Math.floor(random() * 2)', self.html)
        self.assertIn('value="8"', self.html)
        self.assertIn("Arrival times must be non-negative integers.", self.html)

    def test_simulator_tracks_ready_blocked_and_completed_states(self):
        for state in ("Ready", "Running", "Blocked", "Completed"):
            self.assertIn(f'"{state}"', self.html)
        self.assertIn("ioEvents", self.html)
        self.assertIn("process.readyWaiting", self.html)
        self.assertIn("process.ioIntervals", self.html)

    def test_metrics_and_visualization_use_event_intervals(self):
        self.assertIn("waiting: process.readyWaiting", self.html)
        self.assertIn("utilization: 100 * totalCpu / duration", self.html)
        self.assertIn("renderProcessTimelines", self.html)
        self.assertIn("Blocked I/O intervals", self.html)
        self.assertIn("Ready-queue intervals", self.html)
        self.assertIn("process.readyIntervals", self.html)
        self.assertIn("CPU bursts must be a non-empty array", self.html)
        self.assertIn("I/O waits must contain one positive integer", self.html)
        self.assertIn("Compare identical I/O workload", self.html)
        self.assertIn("io-comparison-gantts", self.html)
        self.assertIn("comparison CPU Gantt chart", self.html)

    def test_sjf_round_robin_comparison_reuses_one_input(self):
        self.assertIn("const input = makeIoComparisonWorkload()", self.html)
        self.assertIn('simulate(input, "SJF", quantum)', self.html)
        self.assertIn('simulate(input, "Round Robin", quantum)', self.html)
        self.assertIn("no winner is assumed", self.html)
        self.assertIn("metrics.response.toFixed(2)", self.html)
        self.assertIn('pid: "C1"', self.html)
        self.assertIn('pid: "C6"', self.html)
        self.assertIn("quantum controls Round Robin", self.html)

    def test_comparison_input_is_identical_for_all_policies(self):
        self.assertIn("const input = makeIoComparisonWorkload()", self.html)
        self.assertEqual(self.html.count("simulate(input, \"SJF\", quantum)"), 1)
        self.assertEqual(self.html.count("simulate(input, \"Round Robin\", quantum)"), 1)

    def test_workload_statistics_use_cpu_timeline_not_io_sum(self):
        for expression in (
            "totalCpu",
            "totalIo",
            "cpuBusy",
            "cpuIdle: duration - cpuBusy",
            "utilization: 100 * totalCpu / duration",
            "cpuBusy,",
        ):
            self.assertIn(expression, self.html)
        self.assertIn("Total CPU service", self.html)
        self.assertIn("Total I/O wait", self.html)
        self.assertIn("CPU idle time", self.html)
        self.assertIn("const cpuBusy = timeline.reduce", self.html)
        self.assertIn("const duration = time - start", self.html)

    def test_browser_and_python_benchmark_remain_separate(self):
        self.assertIn("does not call the Python scheduler", self.html)
        self.assertIn("benchmarkRows", self.html)


if __name__ == "__main__":
    unittest.main()
