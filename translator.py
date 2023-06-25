import pandas as pd
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer, AutoTokenizer, RobertaModel

class Translator:
    def __init__(self) -> None:
        super()
        self.m2m100 = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
        self.m2m100tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")

    def translate(self, polish_texts):
        self.m2m100tokenizer.src_lang = "pl"
        encoded_pl = self.m2m100tokenizer(polish_texts, return_tensors="pt", padding=True)
        generated_tokens = self.m2m100.generate(**encoded_pl, forced_bos_token_id=self.m2m100tokenizer.get_lang_id("en"))
        return self.m2m100tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

    def massive_pl_to_en(self, dataloader, filename):
        en_dataset = []
        all_labels = []
        for iteration, (sents, labels) in enumerate(dataloader):
            print(iteration)
            en_dataset += self.translate(sents)
            all_labels += labels.tolist()
        en_dataset = list(zip(en_dataset, all_labels))
        df = pd.DataFrame(en_dataset, columns=['sentence', 'intent'])
        df.to_csv(filename)
        return df