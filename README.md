# pic-classify
Use pretrained resnet-152 for transfer learning on multilabel classification task.

## Data Format  
### Image Directory Structure 

your_img_folder/\
├── img_0001.jpg\
├── img_0002.jpg\
└── ...


### Label CSV File  
`labels.csv` must contain:  
- A `filename` column matching image names  
- Binary columns (`0` or `1`) for each label  

Example:  
| filename      | label_1 | label_2 | ... | label_100 |  
|---------------|---------|---------|-----|-----------|  
| img_00001.jpg | 1       | 0       | ... | 1         |  
| img_00002.jpg | 0       | 1       | ... | 0         |  

# Environments

Python 3.11.13

numpy 2.0.2

torch 2.6.0+cu124

pandas 2.2.2

sklearn 1.6.1

skmultilearn 0.2.0

flask 3.1.1

# Train
Modify `Config` class in `main.py` with your parameters.

DATA_DIR: filepath of dataset

CSV_PATH: filepath of label csv file

NUM_CLASSES: count of labels for your dataset

SAVE_DIR: where to save yor models and checkpoints

### run training:
```
python main.py
```

# Web Interface

A simple local WebUI based on flask is provided in `/reason`.

Change path in config.py and put your `model_name.pth`, `label_list.json` and `model_config.json` to `/reason/model`.

Then update paths in `/reason/config.py`

Run  `app.py` :
```
python app.py
```

WebUI default at http://localhost:5000


# Results
Test on:
- 6000+ imgs 
- 900 labels
- Training time: 10~ hours (on GPU)
