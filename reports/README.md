# Deliverables

All quantitative results were measured locally on the approved 500-article test
sample. No upstream model-card scores are substituted for the experiment.

- `project_report.md`: approach, architecture, model choice, results, learnings.
- `evaluation_report.md`: evaluation protocol, measured ROUGE, intervals, latency.
- `qualitative_review.md`: source-based review of five unselected sample outputs.
- `sample_summaries.md` / `.json`: five full articles, references, generated summaries.
- `metrics.json`: aggregate results, experiment settings, hardware, and code hashes.
- `predictions.jsonl`: all 500 generated outputs and individual ROUGE scores.
- `figures/`: ROUGE interval chart and input-length/latency distributions.
- `environment.txt`: installed package snapshot.
- `unit_tests.xml`, `integration_tests.xml`: pytest results.
- `cpu_smoke_test.json`: actual CPU inference output.
- `http_smoke_test.json`: real HTTP API check results.
- `verification.json`: final checks across all deliverables.

`evaluation_progress.jsonl` is ignored by Git and supports resuming compatible
runs. `predictions.jsonl` is the final tracked result of the complete run.
