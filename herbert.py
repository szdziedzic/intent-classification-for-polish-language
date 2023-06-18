from tqdm import tqdm
from transformers import AutoTokenizer
from transformers import AutoModel
from torch import Tensor
from torch import nn
from torch.optim import Optimizer
import torch.nn.functional as F
from experiment import Experiment
from typing import Union
from massive_dataset import MASSIVE_DATASET_INTENTS
import torch
from sklearn.metrics import accuracy_score

HERBERT_HUGGINGFACE_MODEL_ID = "allegro/herbert-base-cased"
HERBERT_HUGGINGFACE_TOKENIZER_ID = "allegro/herbert-base-cased"


class HerbertTokenizer:
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


class HerbertModel:
    def __init__(self, model: AutoModel, tokenizer: AutoTokenizer):
        super(HerbertModel, self).__init__()
        self.model = model.cuda()
        self.tokenizer = tokenizer

    def __call__(self, x: Tensor) -> Tensor:
        return self.model(**self.tokenizer.tokenize_batch(x)).pooler_output


class HerbertMASSIVEIntentClassifier(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.linear = nn.Linear(hidden_size, len(MASSIVE_DATASET_INTENTS))

    def forward(self, x):
        x = self.linear(x)
        return F.softmax(x)


class HerbertExperiment(Experiment):
    def __init__(
        self,
        optimizer_class: Optimizer,
        loss_class: nn.Module,
        num_epochs: int = 100,
        lr: float = 3e-4,
        test_size: Union[int, None] = None,
        val_size: Union[int, None] = None,
        train_size: Union[int, None] = None,
        neptune_run=None,
        batch_size: int = 32,
    ):
        super(HerbertExperiment, self).__init__(
            optimizer_class=optimizer_class,
            loss_class=loss_class,
            num_epochs=num_epochs,
            lr=lr,
            test_size=test_size,
            val_size=val_size,
            train_size=train_size,
            neptune_run=neptune_run,
            batch_size=batch_size,
        )
        self.name = "Herbert experiment"
        bare_model = AutoModel.from_pretrained(HERBERT_HUGGINGFACE_MODEL_ID)
        bare_tokenizer = AutoTokenizer.from_pretrained(
            HERBERT_HUGGINGFACE_TOKENIZER_ID
        )
        self.model = HerbertModel(bare_model, HerbertTokenizer(bare_tokenizer))
        intent_clf = HerbertMASSIVEIntentClassifier(
            self.model.model.config.hidden_size
        )
        self.intent_clf = intent_clf.cuda()
        self.loss_fn = self.loss_class()
        self.opt = self.optimizer_class(
            self.intent_clf.parameters(), lr=self.lr
        )

    def _train(self) -> None:
        epoch_progress = tqdm(list(range(self.num_epochs)))

        history = []
        for i in epoch_progress:
            train_loss = 0
            y_train_predicted = []
            y_train_true = []
            self.intent_clf.train()
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
