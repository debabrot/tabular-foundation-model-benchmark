# Seoul Bike benchmark artifacts

The [executed regression notebook](../../notebooks/02_regression.ipynb) compares seven
model variants using the same seed-42 random 60/20/20 split of all 8,760 source rows.
Raw and engineered foundation-model settings are labeled separately.

All seven variants completed. Foundation models ran on an NVIDIA GeForce RTX 3050
6GB Laptop GPU; Mitra used bfloat16 autocast. Engineered TabPFN obtained the lowest
observed test RMSE, **104.7439 bikes/hour**, with MAE **48.1243** and R² **0.9729**.
These are descriptive results from this single split, without significance testing.

- [RMSE, MAE, R², and model statuses](full/metrics.csv)
- [RMSE comparison plot](full/rmse.png)
- [Predictions, settings, metrics, and provenance](full/results.json)
- [Source rows, split positions, and protocol](full/splits.json)

This evaluates retrospective tabular prediction with observed weather. It does not
measure future forecasting performance: neighboring hours may span partitions, and
weather forecasts are not modeled. See the [reproduction guide](../../docs/seoul-bike.md)
for source attribution, model settings, limitations, and execution commands.

Scores are measured from model predictions. Blank metrics indicate an unevaluated row
whose status and reason explain its runtime limitation. Optional `smoke/` artifacts use
a separate, explicitly labeled 500-row sample and are not full benchmark results.
