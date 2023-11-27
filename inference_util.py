from transformers import AutoModelForTokenClassification, AutoTokenizer
import re

class PrepareModel:
    def __init__(self, checkpoint_path: str = None, is_colab: bool = False, is_lower_text: bool = True):
        if checkpoint_path is None:
            if is_colab:
                checkpoint_path = "/content/chekpoints_to_upload/albert-base-v2/best_checkpoint"
            else:
                checkpoint_path = "chekpoints_to_upload/albert-base-v2/best_checkpoint"

        self.is_lower_text = is_lower_text
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        self.model = AutoModelForTokenClassification.from_pretrained(checkpoint_path)

    def predict_entity(self, sentence, tag_numb_mapping={'B-MON': 0, 'I-MON': 1, 'O': 2}):
        # Clean sentence
        sentence = re.sub(r"\[([A-Za-z0-9_]+)\]", '', sentence)
        sentence = re.sub(r'[^\w\s]', '', sentence)
        if self.is_lower_text:
            sentence = sentence.lower()

        model = self.model
        tokenizer = self.tokenizer

        tokenized_input = tokenizer(sentence, return_tensors="pt").to(model.device)

        outputs = model(**tokenized_input)

        predicted_labels = outputs.logits.argmax(-1)[0]

        named_entities = [tokenizer.decode([token]) for token, label in
                        zip(tokenized_input["input_ids"][0], predicted_labels) if label != tag_numb_mapping['O']][:-1]

        print("Named Entities", named_entities)
