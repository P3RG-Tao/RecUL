# RecUL: Recommendation Unlearning via BCE-Guided Partitioning
# Datasets
We prepared three commonly used datasets of different scales.

Download the  datasets via the link provided in `Data/Original/.txt`, and preprocess them using the `_data_process.py` script.
# Unlearning
All run scripts used in the paper are named according to the method, backbone, dataset, and other configurations.  
For example, `eraser_mf_amazon.py` corresponds to the **RecUL** method with **MF** as the backbone, applied on the **Amazon** dataset:

```
python eraser_mf_amazon.py --lr 5e-5 --embed_size 128 --batch_size 1024 --part_type 0
```
# Requirements
The code runs well under python 3.8. The required packages are as follows:

pytorch == 1.12.1+cu116

torchvision == 0.11.1+cu116


