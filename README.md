# Intent classification for the Polish language

## General information

Research project for the Natural Language Processing with Deep Learning course at Jagiellonian University (Summer 23).

If you want to learn more about our project's topic go to [Intent_classification_for_Polish_language-proposal.pdf](https://github.com/szdziedzic/intent-classification-for-polish-language/blob/main/papers/Intent_classification_for_Polish_language-proposal.pdf) and read our proposal!

## Authors

- Grzegorz Przybylski (@Zwatotem)
- Paweł Fornalik (@dartespl)
- Szymon Dziedzic (@szdziedzic)

## Research Journal

If you want to stay up to date with our research progress visit our [research journal](https://github.com/szdziedzic/intent-classification-for-polish-language/blob/main/RESEARCH_JOURNAL.md)!

## How to train the model

1. Install all necessarry packages:
    - `torch`
    - `neptune-client`
    - `transformers`
    - `datasets`
    - `tqdm`
2. Run `python --model herbert --num_epochs <num of epochs> --lr <learning rate> --test_size <test dataset size>
--train_size <train dataset size> --val_size <val dataset size> --batch_size <batch size> --num_of_layers <num of layers>`

If you want to use [neptune.ai](https://app.neptune.ai/) to track your experiments, you need to set your API token and project id as environment variables:

- `NEPTUNE_API_TOKEN`
- `NEPTUNE_PROJECT`

If you want to test if your changes work locally you can use [`test_notebook`](https://github.com/szdziedzic/intent-classification-for-polish-language/blob/main/colab_experiments/test_notebook.ipynb) for this purpose.
