# Configuration

Contains JSON configuration files organized in two types:

**1. General Config (e.g., [config_atlas.json](config_atlas.json) and [config_brats2d.json](config_brats2d.json)):**
Defines model, dataset, training parameters, optimizer, loss, and metrics. This must be pass as a parameter to the main.py.


```json
{
  "name": "experiment_name",
  "n_gpu": 1,

  "model": {"name": "UNet3D", "params": {...}},

  "dataset": {
    "name": "ATLAS",
    "root_folder": "/path/to/data",
    "transforms": "config/atlas_transforms.json", <-- path of Transforms Config
    ...
  },
  
  "classes":{"0": name_class_0, "1": name_class_1, ...},
  
  "aggregated_regions": { <-- optional: define aggregated regions for metrics
    "region_name_1": [1, 2],  <-- e.g., combine classes 1 and 2
    "region_name_2": [1, 2, 3]  <-- e.g., combine classes 1, 2, and 3
  },

  "loss": {"name": "DiceLoss", "loss_kwargs": {...}},
  "optimizer": {"name": "Adam", ...},
  "scheduler": {"name": "StepLR", ...},
  
  "metrics": [
    {
      "key": "DSC", <-- identification name of the metric
      "name": "DiceMetric", <-- name of the TorchIO/custom metric class
      "params": { "include_background": false }, <-- params for the metric constructor
      "train": true, <-- if metric must be computed during training phase
      "val": true, <-- if metric must be computed during validation phase 
      "test": true <-- if metric must be computed during test phase
    },
    ...
  ]
}
```

**2. Transforms Config (e.g., [atlas_transforms.json](atlas_transforms.json) and [brats2d_transforms.json](brats2d_transforms.json)):**
Specifies preprocessing and augmentation pipelines using TorchIO transforms (to be passed in the general config under the key dataset.transforms).

```json
{
  "preprocessing": [...],
  "augmentations": [...]
}
```

### Best model selection (`model_best.pth`)
The best checkpoint is selected using the **validation** value of the **first metric** in the `metrics` list (if *--validation* flag is passed to the [main.py](/main.py)).

- The metric being tracked can be either `"<key>_mean"` (mean across classes) or `"<key>_aggregated_mean"` (mean across aggregated regions), controlled by the `--eval_metric_type` parameter
- It is assumed to be **maximized** (higher is better)

For example, if the first entry has `"key": "DSC"`, the trainer tracks either `DSC_mean` or `DSC_aggregated_mean` on the validation set, depending on the selected evaluation metric type.

---

## QaTaCov + Text Embeddings Config

[config_qatacov2d_textemb.json](config_qatacov2d_textemb.json) extends the standard configuration by adding `dataset.text_emb_dir` to point to a directory containing precomputed report embeddings:

```
dataset:
  name: QaTaCovTextEmb
  text_emb_dir: /path/to/Text_Embeddings/BioBERT
```

The dataset expects the embeddings directory to contain:

- `reports_emb.npy`: NumPy array of BioBERT embeddings
- `reports_meta.json`: metadata about the embeddings
- `image_to_report_idx.json`: mapping from image stem to report index

Use [scripts/extract_textemb_biobert.py](../scripts/extract_textemb_biobert.py) or the slurm job in `jobs/` to generate these files.
