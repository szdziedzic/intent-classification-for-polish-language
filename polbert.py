from tqdm import tqdm
from transformers import AutoTokenizer
from transformers import AutoModel
from torch import Tensor
from torch import nn
from torch.optim import Optimizer
from experiment import Experiment
from typing import Union
from massive_dataset import MASSIVE_DATASET_INTENTS
import torch
from sklearn.metrics import accuracy_score

POLBERT_HUGGINGFACE_MODEL_ID = "dkleczek/bert-base-polish-uncased-v1"
POLBERT_HUGGINGFACE_TOKENIZER_ID = "dkleczek/bert-base-polish-uncased-v1"


class PolbertTokenizer:
    def __init__(self, tokenizer: AutoTokenizer):
        self.tokenizer = tokenizer

    def tokenize_batch(self, batch: Tensor) -> Tensor:
        t = self.tokenizer.batch_encode_plus(
            batch,
            padding="longest",
            add_special_tokens=True,
            return_tensors="pt",
        )
        for key in t.keys():
            t[key] = t[key].cuda()
        return t


class PolbertModel:
    def __init__(self, model: AutoModel, tokenizer: AutoTokenizer):
        super(PolbertModel, self).__init__()
        self.model = model.cuda()
        self.tokenizer = tokenizer

    def __call__(self, x: Tensor) -> Tensor:
        output = self.model(**self.tokenizer.tokenize_batch(x))
        hidden_state_mean = torch.mean(output.last_hidden_state, dim=1)
        return torch.cat((output.pooler_output, hidden_state_mean), dim=-1)


class PolbertMASSIVEIntentClassifier(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_of_layers: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        layers = []
        prev_layer_size = hidden_size
        for i in range(num_of_layers - 1):
            layers.append(nn.Linear(prev_layer_size, prev_layer_size // 2))
            prev_layer_size = prev_layer_size // 2
            layers.append(nn.BatchNorm1d(prev_layer_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(prev_layer_size, len(MASSIVE_DATASET_INTENTS)))
        layers.append(nn.Softmax())
        self.seq = nn.Sequential(*layers)

    def forward(self, x):
        return self.seq(x)


class PolbertExperiment(Experiment):
    def __init__(
        self,
        optimizer_class: Optimizer,
        loss_class: nn.Module,
        num_epochs: int = 100,
        num_of_layers: int = 3,
        lr: float = 3e-4,
        test_size: Union[int, None] = None,
        val_size: Union[int, None] = None,
        train_size: Union[int, None] = None,
        neptune_run=None,
        batch_size: int = 32,
        train_base_model: bool = False,
        dropout: float = 0.2,
    ):
        super(PolbertExperiment, self).__init__(
            optimizer_class=optimizer_class,
            loss_class=loss_class,
            num_epochs=num_epochs,
            lr=lr,
            test_size=test_size,
            val_size=val_size,
            train_size=train_size,
            neptune_run=neptune_run,
            batch_size=batch_size,
            num_of_layers=num_of_layers,
            train_base_model=train_base_model,
            dropout=dropout,
        )
        self.name = "Polbert experiment"
        bare_model = AutoModel.from_pretrained(POLBERT_HUGGINGFACE_MODEL_ID)
        bare_tokenizer = AutoTokenizer.from_pretrained(
            POLBERT_HUGGINGFACE_TOKENIZER_ID
        )
        self.model = PolbertModel(bare_model, PolbertTokenizer(bare_tokenizer))
        intent_clf = PolbertMASSIVEIntentClassifier(
            self.model.model.config.hidden_size * 2,
            num_of_layers=self.num_of_layers,
            dropout=self.dropout,
        )
        self.intent_clf = intent_clf.cuda()
        self.loss_fn = self.loss_class()
        opt_params = (
            self.intent_clf.parameters()
            if not train_base_model
            else list(self.model.model.parameters())
            + list(self.intent_clf.parameters())
        )
        self.opt = self.optimizer_class(opt_params, lr=self.lr)

    def _train(self) -> None:
        epoch_progress = tqdm(list(range(self.num_epochs)))

        history = []
        for i in epoch_progress:
            train_loss = 0
            y_train_predicted = []
            y_train_true = []
            self.intent_clf.train()
            if self.train_base_model:
                self.model.model.train()
            for iteration, (X_train, y_train) in enumerate(
                self.train_dataloader
            ):
                y_train = y_train.cuda()
                self.opt.zero_grad()
                output = self.model(X_train)
                y_pred = self.intent_clf(output)
                loss = self.loss_fn(y_pred, y_train)
                loss.backward()
                self.opt.step()
                train_loss += loss.item()
                y_train_predicted.extend(
                    torch.argmax(y_pred, dim=-1).cpu().numpy()
                )
                y_train_true.extend(y_train.cpu().numpy())

            val_loss = 0
            y_predicted = []
            y_true = []

            self.intent_clf.eval()
            if self.train_base_model:
                self.model.model.eval()
            with torch.no_grad():
                for iteration, (X_val, y_val) in enumerate(
                    self.val_dataloader
                ):
                    y_val = y_val.cuda()
                    output = self.model(X_val)
                    y_pred = self.intent_clf(output)
                    loss = self.loss_fn(y_pred, y_val)
                    val_loss += loss.item()
                    y_true.extend(y_val.cpu().numpy())
                    y_predicted.extend(
                        torch.argmax(y_pred, dim=-1).cpu().numpy()
                    )
            train_acc = accuracy_score(y_train_true, y_train_predicted)
            val_acc = accuracy_score(y_true, y_predicted)
            epoch_progress.set_description(
                f"#Epoch: {i}, train loss: {train_loss:.2f}, train_acc: {train_acc:.2f}, val loss: {val_loss:.2f}, val_acc: {val_acc:.2f}"
            )
            history.append(
                {"e": i, "train_acc": train_acc, "val_acc": val_acc}
            )
            if self.neptune_run:
                self.neptune_run["train/loss"].log(train_loss)
                self.neptune_run["train/acc"].log(train_acc)
                self.neptune_run["val/loss"].log(val_loss)
                self.neptune_run["val/acc"].log(val_acc)

    def _test(self) -> None:
        test_loss = 0
        y_predicted = []
        y_true = []
        self.intent_clf.eval()
        if self.train_base_model:
            self.model.model.eval()
        with torch.no_grad():
            for iteration, (X_test, y_test) in enumerate(self.test_dataloader):
                y_test = y_test.cuda()
                output = self.model(X_test)
                y_pred = self.intent_clf(output)
                loss = self.loss_fn(y_pred, y_test)
                test_loss += loss.item()
                y_true.extend(y_test.cpu().numpy())
                y_predicted.extend(torch.argmax(y_pred, dim=-1).cpu().numpy())
        test_acc = accuracy_score(y_true, y_predicted)
        print(f"Test loss: {test_loss:.2f}, test_acc: {test_acc:.2f}")
        if self.neptune_run:
            self.neptune_run["test/loss"] = test_loss
            self.neptune_run["test/acc"] = test_acc

    def maybe_safe_classsidier_layer_state_dict_to_neptune_run(self) -> None:
        if self.neptune_run:
            torch.save(self.intent_clf, "weights.pt")
            self.neptune_run["model/intent_clf/weights"].upload(
                "model_checkpoints/weights.pt"
            )
