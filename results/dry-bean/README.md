# Dry Bean benchmark artifacts

The [executed multiclass notebook](../../notebooks/03_multiclass_classification.ipynb) compares
seven model variants on all 13,611 source rows using a common seed-42 stratified split.
Raw and engineered foundation-model settings are labeled separately.

All seven variants completed. Foundation models used an NVIDIA GeForce RTX 3050 6GB
Laptop GPU; Mitra used bfloat16 autocast. Raw TabPFN obtained the highest observed
test macro-F1 (**0.9390**), with weighted-F1 **0.9264** and accuracy **0.9266**.
These are descriptive results from this split, without significance testing.

- [Macro-F1, weighted-F1, accuracy, and model statuses](full/metrics.csv)
- [Macro-F1 comparison plot](full/macro_f1.png)
- [Per-class precision, recall, F1, and support](full/per_class_metrics.csv)
- [Confusion matrices with explicit original-label order](full/confusion_matrices.json)
- [Predictions, configuration, and provenance](full/results.json)
- [Source rows, splits, and protocol](full/splits.json)

Scores are generated from real predictions. Empty metrics indicate unevaluated rows,
with status and runtime reasons recorded explicitly. This is a single-split descriptive
comparison, without significance testing. All source rows are retained, including 68
identical records that limit independence checks. See the
[reproduction guide](../../docs/dry-bean.md) for attribution, methodology, and execution.

Optional `smoke/` artifacts contain explicitly labeled 500-row execution checks and
must not be presented as full-data benchmark results.
