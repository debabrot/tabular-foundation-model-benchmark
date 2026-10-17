# Online Shoppers benchmark artifacts

The [executed binary-classification notebook](../../notebooks/01_binary_classification.ipynb)
compares the four core model families on a common, fixed stratified split of the full
UCI Online Shoppers dataset, including separate raw/engineered foundation-model settings.

All seven variants completed on all 12,330 source rows with seed 42 and a 60/20/20
split. Foundation models used an NVIDIA GeForce RTX 3050 6GB Laptop GPU; Mitra used
bfloat16 autocast. Raw Mitra obtained the highest observed test ROC-AUC (0.9366),
while raw TabPFN obtained the highest average precision (0.7593). These are descriptive
results from one split, without significance testing.

- [Test metrics and model statuses](full/metrics.csv)
- [ROC-AUC plot](full/roc_auc.png)
- [Average-precision plot](full/average_precision.png)
- [Predictions, metrics, configuration, and provenance](full/results.json)
- [Source rows and split positions](full/splits.json)

Blank metrics indicate an unevaluated model; the status and reason explain its runtime
limitation. Results are measured, with no substitute scores. Source attribution,
methodology, hardware requirements, and reproduction commands are in the
[notebook guide](../../docs/online-shoppers.md).

An optional `smoke/` subdirectory contains explicitly labeled 500-row execution checks,
which are separate from the full-data comparison.
