# Traffic Flow Prediction
Traffic Flow Prediction with Neural Networks(SAEs、LSTM、GRU).

## Requirement
- Python 3.6    
- Tensorflow-gpu 1.5.0  
- Keras 2.1.3
- scikit-learn 0.19

## Train the model

**Run command below to train the model:**

```
python train.py --model model_name
```

You can choose "lstm", "gru" or "saes" as arguments. The ```.h5``` weight file was saved at model folder.


## Experiment

Data are obtained from the Caltrans Performance Measurement System (PeMS). Data are collected in real-time from individual detectors spanning the freeway system across all major metropolitan areas of the State of California.
	
	device: Tesla K80
	dataset: PeMS 5min-interval traffic flow data
	optimizer: RMSprop(lr=0.001, rho=0.9, epsilon=1e-06)
	batch_szie: 256 


**Run command below to run the program:**

```
python main.py
```


## Copyright
See [LICENSE](LICENSE) for details.
