# CGI-HRDC 2023 - Hypertensive Retinopathy Diagnosis Challenge
our solution to Task 1: Hypertension classification
before starting, don't forget to load `1-Images`  and `2-Groundtruths` folder to current directory.
our result is presented in `DenseNet` branch.
## train the model
suppose you want to train `net_you_wanna_train`. run in terminal:
```
python3 net_you_wanna_train/main.py
```
## verify the model
`demo/model.py` provides required interface for submission, and also serves as verification script. To verify the model, run:
```
python3 demo/model.py
```
