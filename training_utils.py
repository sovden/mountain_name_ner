import numpy as np
from seqeval.metrics import f1_score, precision_score, recall_score, classification_report
import torch

def compute_metrics(predictions, label_list):
  predictions, labels = predictions
  predictions = np.argmax(predictions, axis=2)

  true_predictions = [
      [label_list[p] for (p,l) in zip(prediction, label) if l != -100]
      for prediction, label in zip(predictions, labels)
  ]

  true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

  return {
        "precision": precision_score(true_labels, true_predictions),
        "recall": recall_score(true_labels, true_predictions),
        "f1": f1_score(true_labels, true_predictions),
        # "classification_report": classification_report(true_labels, true_predictions),
    }


def tokenize_and_align_labels(examples, tokenizer):
    # TODO: READ ABOUT PARAMS HERE
    tokenized_inputs = tokenizer(
        examples["tokens"], truncation=True, is_split_into_words=True, padding=True
    )

    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        # print(word_ids)
        # print(label)
        previous_word_idx = None
        label_ids = []

        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)

            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


def data_collator(data, tokenizer):
    input_ids = [torch.tensor(item["input_ids"]) for item in data]
    attention_mask = [torch.tensor(item["attention_mask"]) for item in data]
    labels = [torch.tensor(item["labels"]) for item in data]


    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)


    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }