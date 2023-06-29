# Taylorformer

Our model architecture is shown below:

<img width="784" alt="image" src="https://github.com/oremnirv/ATP/assets/54116509/7a8f1e82-4f91-4cb2-89ec-748f8556529a">


## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```
## Training and Evaluation

### Training

To train the model(s) in the paper, run this command:

```train
python training_and_evaluation.py "<type of dataset>" "<model>" num_iterations num_repeat_runs n_C n_T 0
```
where <type of dataset> is for example ETT or exchange, <model> is for example, TNP or taylorformer, where n_C and n_T are the number of context and target points, respectively.
  
You will have needed to create appropriate folders to store the model weights and evaluation metrics. We have included a folder for the taylorformer on the ETT dataset, with n_T = 96, as an example. Its path is `weights_/forecasting/ETT/taylorformer/96`.

### Evaluation 

Evaluation metrics (mse and log-likelihood) for each of the repeat runs are saved in the corresponding folder e.g. `weights_/forecasting/ETT/taylorformer/96`. The mean and standard deviations are used when reporting the results.
  
### Load pre-trained model 

 Here is an example of how to load a pre-trained model for the ETT dataset with the Taylorformer for the target-96-context-96 setting.
  
  
```
python pre_trained_model_ex.py 0 37
```
  
## Results
  
We show our results on the forecasting datasets. More results can be found in the paper.
  
<img width="717" alt="image" src="https://github.com/oremnirv/ATP/assets/54116509/45c9efad-41cb-4ad1-aa16-d643eb8e23ad">

  



