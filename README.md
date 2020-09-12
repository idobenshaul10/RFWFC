# Smoothness Analysis on Deep Learning Architectures
**Simple commands** - for questions - ido.benshaul@gmail.com
## Training a model 
```python .\train\train_mnist.py```
## Running Smoothness Analysis
### Specific Epoch:
``` python .\DL_Layer_Analysis\DL_smoothness.py --checkpoint_path "C:\projects\RFWFC\results\DL_layers\trained_models\MNIST\weights.0.h5" --output_folder "C:\projects\RFWFC\results\DL_layers\analysis\results\test" --calc_test --high_range_epsilon 0.1 --env_name mnist```

### Run on Batch:
```python .\DL_Layer_Analysis\run_DL_smoothness_batch.py --env_name mnist --checkpoints_dir_path "C:\projects\RFWFC\results\DL_layers\trained_models\MNIST" --output_folder "C:\projects\RFWFC\results\DL_layers\analysis\results\mnist" --calc_test --high_range_epsilon 0.1```

**Note:** In both scripts - inorder to cluster, add the ```--use_clustering``` flag.
### Plotting Results:
```python .\DL_Layer_Analysis\plot_DL_json_results.py --main_dir C:\projects\RFWFC\results\DL_layers\analysis\results\test --plot_test```




