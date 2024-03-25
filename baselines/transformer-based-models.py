# Author   : Oguzhan Ozcelik
# Date     : 19.08.2022
# Subject  : Transformer-based models for text classification
# Framework: Hugging Face Trainer

import os
import random
import numpy as np
import pandas as pd

from datasets import Dataset, DatasetDict, load_metric
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer, DataCollatorWithPadding)
from transformers import logging
from transformers.trainer_utils import IntervalStrategy
from sklearn.metrics import classification_report

logging.set_verbosity_error()


class TextClassification:
    def __init__(self, train_path, test_path, model_name, num_labels):
        self.train_path = train_path
        self.test_path = test_path
        self.model_name = model_name
        self.num_labels = num_labels
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=self.num_labels)
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

    def data_loader(self):
        train_pd = pd.read_csv(self.train_path, sep='\t', header=0, encoding='utf-8')
        test_pd = pd.read_csv(self.test_path, sep='\t', header=0, encoding='utf-8')

        train_pd = train_pd.dropna()
        test_pd = test_pd.dropna()

        train_dataset = Dataset.from_pandas(train_pd[['text', 'label']])
        test_dataset = Dataset.from_pandas(test_pd[['text', 'label']])

        return train_dataset, test_dataset

    def preprocess_function(self, examples):
        return self.tokenizer(examples["text"], truncation=True, padding="longest", max_length=128)

    def compute_metrics(self, eval_pred):
        metric1 = load_metric("precision")
        metric2 = load_metric("recall")
        metric3 = load_metric("f1")
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        precision = metric1.compute(predictions=predictions, references=labels, average="weighted")["precision"]
        recall = metric2.compute(predictions=predictions, references=labels, average="weighted")["recall"]
        f1 = metric3.compute(predictions=predictions, references=labels, average="weighted")["f1"]
        return {"precision": precision, "recall": recall, "f1": f1}

    def train(self, train_dataset, test_dataset):
        dataset = DatasetDict()
        dataset['train'] = train_dataset
        dataset['test'] = test_dataset
        dataset = dataset.map(self.preprocess_function, batched=True)

        training_args = TrainingArguments(
            output_dir="./results",
            learning_rate=5e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=10,
            report_to=["none"],
            logging_strategy=IntervalStrategy.STEPS,
            save_strategy=IntervalStrategy.NO,
            seed=random.randint(1, 1000),
            disable_tqdm=False,
            logging_steps=50
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset["train"],
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics,
        )

        trainer.train()
        pred = trainer.predict(dataset["test"].remove_columns(["label"]))
        print(classification_report(dataset["test"]["label"], pred.predictions.argmax(axis=1), digits=4))
        return dataset["test"]["label"], pred.predictions.argmax(axis=1)


if __name__ == "__main__":
    models = [
        ["google-bert/bert-base-uncased", "bert"],
        ["microsoft/deberta-v3-base", "deberta"],
        ["google-bert/bert-base-multilingual-uncased", "mbert"],
        ["FacebookAI/xlm-roberta-base", "xlm-r"],
        ["dbmdz/bert-base-turkish-uncased", "berturk"]
    ]

    for lang in ['EN', 'TR']:
        print(f"Language: {'English' if lang == 'EN' else 'Turkish'}")

        for model in models:
            print('Model: ', model[0])
            path = os.path.join('results', model[1], lang)
            if not os.path.exists(path):
                os.makedirs(path)

            for fold in range(5):
                print(f"Fold: {fold}")

                train_object = TextClassification(
                    train_path=os.path.join('./dataset', lang, 'folds', lang+'_train_'+str(fold)+'.tsv'),
                    test_path=os.path.join('./dataset', lang, 'folds', lang+'_test_'+str(fold)+'.tsv'),
                    model_name=model[0],
                    num_labels=3)

                train_data, test_data = train_object.data_loader()

                y_true, y_pred = train_object.train(train_dataset=train_data, test_dataset=test_data)
                report = classification_report(y_true, y_pred, digits=4)
                print(report)

                with open(os.path.join(path, 'classification_report_' + str(fold)), 'w') as file:
                    file.write(report)