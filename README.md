# AI/ML-Based Adaptive CPU Scheduling Algorithm for Dynamic Workload Optimization

**Course:** Operating System (BCSE303P)  
**Team:** Aditya Avva (24BAI0048), Ribhav Aggarwal (24BAI0246)  
**Faculty:** Dr. Anisha M. Lal

## Windows setup and run

Run these commands in **Command Prompt** from the project folder. Python 3.10 or newer is required.

1. Confirm Python is available:
   ```bat
   py --version
   ```
2. Install Matplotlib for the Python benchmark comparison charts:
   ```bat
   py -m pip install matplotlib
   ```
3. Run the default reproducible hand-designed benchmark:
   ```bat
   py src\adaptive_scheduler.py
   ```
4. Choose the recorded seed and Round Robin quantum. The hand-designed scenarios
   always contain five explicitly documented processes:
   ```bat
   py src\adaptive_scheduler.py --seed 17 --quantum 3
   ```
5. To view the interactive browser demo and allow it to read the benchmark CSV, start
   Python's local web server from the project folder:
   ```bat
   py -m http.server 8000
   ```
   Open `http://localhost:8000/website/` in a browser. Keep the server window open
   while presenting. Opening `website\index.html` directly also shows the simulator
   and checked-in chart images, but browsers commonly block CSV fetches from `file:`.
6. Or double-click `run_project.bat` to run the default Python experiment.

The program writes `results\benchmark_results.csv`, `results\benchmark_results.json`,
`results\avg_waiting_comparison.png`, and `results\avg_turnaround_comparison.png`.
Run it from the project folder so the `results` directory is created there. Matplotlib
is used only for the Python-generated charts; scheduling and CSV/JSON export use the
standard library.

Run the regression tests from the project folder:
```bat
py -m unittest discover -s tests -v
```

## Browser demo and Python benchmark

`website\index.html` is a standalone browser demonstration. It includes a separate
deterministic process generator, local event-driven scheduling simulation, process
metrics, and a Gantt chart. It does not invoke Python. When served from the project folder with
`py -m http.server 8000`, it fetches `results\benchmark_results.csv` to render
comparison charts. Refreshing the page fetches the latest CSV; the **Refresh benchmark
results** button fetches it again without reloading the page. A failed fetch displays an
error and clears the benchmark charts instead of showing stale values. Each chart
highlights the lowest average for each workload. The benchmark section also displays
the seed, process count, and Round Robin quantum when those columns exist in the CSV.
Use the benchmark-scenario selector to compare `long-job-convoy`, `urgent-priority`,
`staggered-response`, and `balanced-mix`. Each is a documented, deterministic process
table rather than a random sample. All four algorithms receive exactly the same process
inputs within a scenario. The browser simulator remains separate from these Python
benchmark inputs.
Changing browser simulator controls updates only its simulation; it does not change the
Python benchmark charts. Its process inputs are separate from the authoritative Python
benchmark results.

## Algorithms

- **FCFS (First Come, First Served):** Runs processes in arrival order without preemption.
- **SJF (Shortest Job First):** Among arrived processes, runs the process with the
  shortest CPU burst; it is non-preemptive.
- **Priority Scheduling:** Among arrived processes, runs the process with the
  numerically smallest priority; it is non-preemptive.
- **Round Robin:** Runs ready processes in turn for at most the configured time quantum,
  returning unfinished processes to the ready queue.

The Python scheduler is a discrete-time CPU-service simulation. Its benchmark process
model remains one CPU burst per process, so the verified Python benchmark scenarios are
unchanged. The browser simulator deliberately has a richer demonstration-only model:
each process can alternate between multiple CPU bursts and simulated I/O waits.
The simulated scheduling order is not a measurement of how an actual Windows or Linux
kernel schedules programs.

## Workloads and adaptive policy

The browser generator creates three labelled synthetic workload categories:

- CPU-intensive: predominantly long CPU bursts with infrequent or short I/O waits.
- I/O-intensive: short CPU bursts separated by frequent, longer simulated I/O waits.
- Mixed: alternating longer and shorter CPU bursts with intermediate I/O waits.

When a browser process finishes a non-final CPU burst, it enters `Blocked` for its
configured I/O duration and returns to `Ready` when that interval ends. FCFS, SJF,
Priority, and Round Robin all use the same event-driven model. I/O waits are simulated
time intervals only; there is no real device operation, blocking system call, or kernel
modification.

The adaptive component is a small **nearest-centroid classifier**. It is trained on
labelled examples generated from the same three synthetic burst distributions, then
classifies browser workloads using mean CPU-burst length, CPU-burst standard deviation,
the fraction of short CPU bursts, and the ratio of simulated I/O wait to CPU service. The predicted class is
mapped transparently to a policy:

| Predicted workload | Selected policy |
|---|---|
| CPU-intensive | SJF |
| I/O-intensive | Round Robin |
| Mixed | Priority |

This is a genuine classification-and-selection step, but it is an educational model on
synthetic data. The program does not report a measured ML accuracy, and the policy is
not guaranteed to outperform every fixed scheduler. Its training examples use a fixed
seed (`123`); `--seed` controls the generated benchmark workloads.

## Metrics and benchmark interpretation

Each algorithm is run on the same hand-designed process set for each scenario. The
default benchmark writes one row per scenario/algorithm combination
(4 scenarios x 4 algorithms = 16 rows). The older random scenarios remain available
through `generate_scenario` for compatibility, but are not the authoritative dashboard
benchmark.
Each row is marked `is_adaptive` when it is the classifier-selected policy. The chosen
policy is compared against the other three policies; it is not an additional fifth
algorithm run.

The output also includes automatically measured winner summaries for waiting time,
turnaround time, response time, highest-priority response time, throughput, and
utilization. The highest-priority metric uses every process tied at the smallest
numeric priority value; lower average response is better. Winners are derived from
the four measured rows in each scenario; equal values are reported as ties. The
scenario definitions do not hard-code metric winners or claim that one policy is
universally best.

The hand-designed scenarios are:

- `long-job-convoy`: a long process and short processes are ready together, exposing
  the convoy effect and SJF's short-job advantage.
- `urgent-priority`: priority `1` is urgent, including a later-arriving tied urgent
  process. This tests the separate response metric for the highest-priority group.
- `staggered-response`: arrivals at several time points make Round Robin's first
  response and fairness visible.
- `balanced-mix`: short CPU bursts with varied arrivals and priorities. It is labelled
  I/O-intensive for comparison only; it does not simulate blocking I/O.

- **Waiting time:** Total time spent in the `Ready` state. Time spent `Blocked` for
  simulated I/O is excluded.
- **Turnaround time:** Completion time minus arrival time.
- **Response time:** First CPU start minus arrival time.
- **Throughput:** Number of completed processes divided by elapsed time from the first
  arrival to the last completion.
- **CPU utilization:** Total CPU service demand across all CPU bursts divided by that
  same elapsed workload window. Initial idle time before the first arrival is excluded;
  idle gaps caused by arrivals or I/O blocking reduce utilization.

The browser process table shows final state, CPU-burst sequence, and I/O-wait sequence.
Its event timeline lists CPU execution and blocked intervals separately, while the Gantt
chart shows CPU execution and idle periods. The browser metrics are not mixed with the
Python benchmark CSV metrics.

The default browser I/O-intensive workload uses 5–8 processes (the control defaults to
8), several CPU bursts of about 1–3 time units, and varied simulated I/O waits of
10–15 time units. With the default seed, arrivals are deterministic and sampled across
time 0–20 rather than clustered at the start; this produces both early and noticeably
later arrivals. This creates a natural
period in which the initial ready work has finished and all remaining processes are
blocked or have not arrived, so the CPU can become idle without an artificial pause.
Other processes still run while individual processes are blocked.

For the default eight-process run with seed `42`, the sorted arrival times observed in
the browser are `1, 1, 5, 10, 12, 13, 13, 14`. The same generated process records are
used for every selected scheduling algorithm; only scheduling order changes.

The browser also reports total CPU service, total requested I/O wait, average CPU-burst
length, CPU bursts per process, CPU busy time, CPU idle time, utilization, and
throughput. CPU utilization is calculated as:

`CPU busy time / (final completion - first arrival) * 100`

I/O waits are not summed into CPU busy time; waits can overlap. Ready-queue waiting is
tracked separately from blocked I/O time. The per-process timeline shows CPU execution,
ready-queue intervals, and blocked intervals.

The browser includes a **Fair I/O comparison: SJF vs Round Robin** control. It runs
both policies on the same deterministic six-process I/O-intensive input, including the
same arrivals, CPU-burst sequences, I/O waits, and priorities. The configured Round
Robin quantum is used for that comparison, and both measured metric rows and CPU Gantt
charts are shown. This comparison is intentionally descriptive rather than a claim that
Round Robin must win.

Round Robin can improve average response time because each ready process receives an
early time slice instead of waiting for another process's full CPU burst. It may still
have higher ready-queue waiting or turnaround time than SJF: frequent requeueing and
fair sharing can delay completion, while SJF favors short available bursts. The
comparison therefore reports waiting time, turnaround time, response time, and
utilization from the event timeline instead of assuming a winner.

`makespan` in the exports is the final completion timestamp; `simulation_duration` is
the elapsed workload window used for throughput and utilization. These are simulation
metrics, not measurements of machine CPU usage.

The benchmark records the requested seed, classifier seed, fixed process count (five),
quantum so a run's configuration is visible. The hand-designed process inputs do not
change when `--seed` changes; this makes the scenario definitions reproducible and
inspectable. The benchmark also records serialized process inputs, scenario
description, and winner fields. The Python charts use the first hand-designed
scenario; the website can display any scenario from the CSV.

## Project files

- `src\adaptive_scheduler.py` — workload generation, classifier, scheduler, metrics,
  exports, and chart generation.
- `tests\test_adaptive_scheduler.py` — focused scheduler and output regression tests.
- `results\` — CSV/JSON benchmark outputs and generated comparison charts.
- `docs\Project_Report.docx` — project report.
- `docs\Project_Presentation.pptx` — project presentation.
