# Dataset

Approved dataset: [CNN/DailyMail](https://huggingface.co/datasets/abisee/cnn_dailymail),
configuration **3.0.0**, official **test** split (11,490 rows).
Each record has `id`, `article`, and human-written `highlights`.

We sample 500 rows without replacement using NumPy's default_rng with seed 42.
The selected row indices and IDs are saved in `evaluation_manifest.json`.
Only the approximately 30 MB test Parquet file is downloaded. The training
split is not used. Data and model revisions are pinned in local manifests.

The Hugging Face repository metadata labels the dataset Apache-2.0; its card's
license paragraph specifically discusses version 1.0.0. Original news articles
remain attributable to their publishers. Preserve dataset and paper citations;
the repository does not relicense underlying publisher content.

Raw data and the evaluation subset are ignored by Git and reproduced by the
preparation command. The report contains a small set of attributed examples.

Reference: See, Liu, and Manning (2017),
[Get To The Point: Summarization with Pointer-Generator Networks](https://aclanthology.org/P17-1099/).
