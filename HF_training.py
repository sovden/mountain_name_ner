from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer,  TrainingArguments
from datasets import Dataset, DatasetDict

import functools

from data_preparation.data_preparation import GetFormattedData, check_data_for_consistency
from training_utils import data_collator, compute_metrics, tokenize_and_align_labels

def create_datasets(is_colab=False):
    formatted_data = GetFormattedData(is_colab=is_colab)
    train_set_dict, val_set_dict, test_set_dict = formatted_data.get_split_formatted_data()

    check_data_for_consistency(train_set_dict)
    check_data_for_consistency(val_set_dict)
    check_data_for_consistency(test_set_dict)

    train_dataset_custom = Dataset.from_dict(train_set_dict)
    val_dataset_custom = Dataset.from_dict(val_set_dict)
    test_dataset_custom = Dataset.from_dict(test_set_dict)

    datasets_custom = DatasetDict({
        "train": train_dataset_custom,
        "validation": val_dataset_custom,
        "test": test_dataset_custom
    })

    label_list = formatted_data.label_list

    return datasets_custom, label_list


def run_training(is_colab=False,
                 save_name=None,
                 model_name="albert-base-v2",
                 num_train_epochs=1,
                 learning_rate=5e-5,
                 scheduler="cosine",
                 dataset_params=None):

    if save_name is None:
        save_name = model_name

    print(f"run: {save_name}")

    if dataset_params:
        datasets_custom, label_list = dataset_params
    else:
        datasets_custom, label_list = create_datasets(is_colab)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=len(label_list))

    tokenized_datasets = datasets_custom.map(functools.partial(tokenize_and_align_labels, tokenizer=tokenizer), batched=True)

    training_args = TrainingArguments(
        output_dir=f"./results/{save_name}",
        evaluation_strategy="steps",
        eval_steps=200,
        save_steps=200,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        logging_steps=100,
        learning_rate=learning_rate,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
    )

    if scheduler:
        training_args = training_args.set_lr_scheduler(name=scheduler, warmup_ratio=0.05)

    # get_compute_metrics = CreateComputeMetrics(label_list=formatted_data.label_list).compute_metrics
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=functools.partial(data_collator, tokenizer=tokenizer),
        tokenizer=tokenizer,
        compute_metrics=functools.partial(compute_metrics, label_list=label_list),
    )

    trainer.train()
    trainer.evaluate(tokenized_datasets["test"])
