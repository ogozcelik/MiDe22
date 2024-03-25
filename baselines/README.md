# Source Codes of the Baseline Models

## Environment
- Python 3.8.10
- transformers 4.36.2
- scikit-learn 1.3.2
- pandas 2.0.3
- numpy 1.24.3  
- torch 1.13.1
- torchdata 0.7.1       
- torchtext 0.7.0 

Run the example command below to train and test the models. Please note that before running the command map the tweet texts in each fold, as an additional `text` column to the dataframes.

```bash
$ python3 svm.py
$ python3 lstm-and-bilstm.py
$ python3 transformer-based-models.py
```
