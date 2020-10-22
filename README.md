# Time Series Forecasting


## Kaggle Project for Machine learning course at Mines Nancy school of engineering.
Predict Future Sales
Final project for "How to win a data science competition" Coursera course

## Training of the model

To install the prerequisites into the pytorch_lab conda environment, run

```
conda pytorch_lab_env create -f pytorch_lab.yml
```

The main code is in the Jupyter Notebook but also converted in PY files if needed.

## Analysis

Of course my model is not really representing the reality and is not complete, I would normally need to create a model by item-shop duo and then connect them all together for example. To work with a bigger dataset and add additional values I could also use a k-fold technique. Maybe convolutional layers are also not the best way to predict this kind of information. Overall, the Loss looks good but I'm not sure it would with a different dataset of values.

If I had more time here's what I wouldve done:

- Build a model for each item-shop duo
- Connect them all with a fully connected layer
- Build a model by shop (for all items)
- Build a model by item (for all shops)

I could've compare all those models and select my favorite one or analyse them too.
