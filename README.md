# Weather_Forecast_in_HCM

## Requirements
```python
pip install -r requirements.txt
```

## Train
### Here is some arguments
###
### --num_epochs
### --num_workers
### --lr
### --pin_memory
### --weight_decay
###
### Here is example use
 ```python
python train.py --num_epochs 25
```

## Test
### Here is some arguments
###
### --device
### --model_name
###
### Here is example use
 ```python
python test.py --model_name epochs/pangu_lite_epoch_25.pth
```