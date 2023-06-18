# Research journal for the "Intent classification for the Polish language" project

## 2023-05-19

- We were working on the proposal for our research project.
- The topic was chosen and we started writing the proposal.

## 2023-05-20

- We finished and submitted our proposal.

## 2023-06-17

- Loaded the MASSIVE dataset with its Polish subset.
- Successfully loaded the Herbert model using the `huggingface/transformers` library.
- Added a linear classification layer to perform intent classification using representations from Herbert.
- We created and ran a training loop to ensure that our linear layer is learning on a small subset of the dataset and we plan to run it on a whole dataset.

## 2023-06-18

- Move our model to the `py` files and refactor the code.
- Monitor the model learning process using neptune.ai.
- Ran experiments on the whole dataset.
- Accuracy for 1 linear layer wasn't great (40%) so we decided to add another linear layer.
- Experiment with different hyperparameters and normalization techniques to obtain the best results.
- Experiment with different architectures of the classifier
- Try both transfer learning and finetuning of the base model as well
