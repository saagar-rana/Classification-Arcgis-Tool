# CHM regression models

Both **Train CHM Regressor** (`Unet_Segmentor.pyt`) and **Predict CHM from Imagery**
(`Inference.pyt`) offer these Model Type options:

| Model Type | Definition | Architecture |
| --- | --- | --- |
| `UNet` | Existing definitions in the toolboxes | Original U-Net, 64 starting channels; existing checkpoints remain compatible. |
| `ResUNet` | `resunet.py` | Residual encoder and decoder blocks, four pooling stages, 32 starting channels. |
| `UNet++` | `unetplusplus.py` | Five resolution levels with nested dense skip connections, bilinear upsampling, 32 starting channels, one final regression head without deep supervision. |

The new architectures are compact CHM adaptations. All three models use four input
bands, one output band, and a ReLU output so predicted heights are nonnegative.
The new models also match decoder features to skip dimensions when tile sizes are odd.

## Training and prediction

1. Refresh the training toolbox in ArcGIS Pro, then select `ResUNet` or `UNet++`
   under **Model Type**. Existing training parameters, epoch metrics, early
   stopping, and optional test folders apply to either model.
2. Save each experiment to a separate `.pth` filename. The adjacent
   `.metadata.json` records the selected architecture and its configuration.
3. For prediction, refresh the inference toolbox and select the **same Model Type**
   used during training, then choose its `.pth` file. Keep the metadata beside
   the weights so mismatched architecture or preprocessing settings can be detected.
4. Optional pretrained weights must match the selected model exactly. Loading
   initializes model weights; the optimizer and epoch counter start fresh.

`chm_models.py` supplies the shared model selection and strict weight loader.
Keep it, `resunet.py`, and `unetplusplus.py` beside both `.pyt` toolboxes when copying
the tools to another location. Models load by absolute sibling paths, so no
Python package installation or manual Python path changes are needed. They use
PyTorch already required by the existing toolboxes.

## Data and comparison

Imagery is always divided by **255** during training and prediction. Pixels where
all four imagery bands are zero are treated as padding and excluded from the
loss and reported metrics. Valid zero-height ground remains part of training.
CHM labels retain their source height units; prediction NoData is `-9999`.

Start with `ResUNet`, then compare `UNet++` with the existing `UNet`. Use the same
geographically separate test zone and compare test MAE, RMSE, and bias. The
training tool currently generates a fresh random 80/20 development split on each
run, so repeated runs do not guarantee identical training/validation membership.
Architecture size alone does not establish prediction quality or training speed.
