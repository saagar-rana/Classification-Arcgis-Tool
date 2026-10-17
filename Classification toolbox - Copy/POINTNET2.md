# PointNet++ in the ArcGIS Pro toolbox

Select **PointNet++** under **Model Type** in Classify Trees. Refresh the toolbox or restart ArcGIS Pro after updating the files.

This integration uses single-scale grouping (SSG), implemented in pointnet2_torch.py with standard PyTorch. It does not require pointnet2_ops. The original models/pointnet2 research implementation remains available separately and is not the backend used by this selector. Keep pointnet2_torch.py beside model_adapter.py and the .pyt file.

## Inputs and settings

- Input features: XYZ coordinates, RGB, and intensity, in that order.
- XYZ drives farthest-point sampling and radius grouping; RGB and intensity supply four additional features.
- Two local abstraction stages sample up to 512 and 128 centroids with up to 32 and 64 neighbors, followed by global abstraction and a classification head.
- Radius defaults: 0.5 and 1.5 in the source coordinate units. For metric clouds these are metres. Verify that these neighborhoods fit the tree density and scale; coordinates retain their original scale after training preprocessing.
- Default batch size: 4 when the user has not manually set it. Points per Tree remains configurable, defaulting to 4096.
- Seven species: 1,464,135 trainable parameters.

Use the existing split CSV, epoch, patience, and output-weight controls. Best checkpoint metadata records the PointNet++ name, input channels, centroid counts, neighborhood radii, neighborhood sizes, dropout, and implementation version. The polygon prediction loader recognizes these checkpoints and reconstructs their saved configuration.

## Performance and validation

This standard PyTorch backend favors compatibility. Its iterative farthest-point sampling can be slower than optimized compiled CUDA PointNet++ operators. Training performance and species accuracy must be measured on the actual dataset.

Checks performed: syntax, seven-channel forward/backward with finite gradients, small point clouds, checkpoint round trip, and CUDA inference on 4096-point trees. ArcGIS GUI execution and full-dataset accuracy remain to be verified.
