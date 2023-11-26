import random
import os


def read_conll_like_file(file_path: str) -> list:
    with open(file_path, "r") as f:
        content = f.read().strip()
        sentences = content.split("\n\n")
        data = []
        for sentence in sentences:
            tokens = sentence.split("\n")
            token_data = []
            for token in tokens:
                token_data.append(token.split())
            data.append(token_data)
    return data


def check_data_for_consistency(formatted_data: dict) -> None:
    issue_count = 0
    for i in range(len(formatted_data["tokens"])):
        if len(formatted_data["tokens"][i]) != len(formatted_data["ner_tags"][i]):
            issue_count += 1
            print(issue_count, len(formatted_data["tokens"][i]), len(formatted_data["ner_tags"][i]))
            print(formatted_data["tokens"][i], formatted_data["ner_tags"][i])

    if issue_count == 0:
        print("number of tokens = number of tags for each sentence")


class GetFormattedData:
    def __init__(self, is_colab: bool = False):
        if is_colab:
            root_dir = "/content/data_preparation/"
        else:
            root_dir = ""

        self.conll_train = os.path.join(root_dir, "clean_data/conll2003/eng.train")
        self.dataset_mountain_path = os.path.join(root_dir, "clean_data/dataset_mountains.txt")

        self.label_list = None
        self.label_map = None

    @staticmethod
    def formate_conll_data_for_adding(data: list) -> dict:
        """
    Create {"tokens": [], "ner_tags": []} based on the conll2003 like raw data,
    where all ner_tags ~ 'O' (no entity)

    :param data: conll2003 like raw data
    :return:  {"tokens": [], "ner_tags": []}
    """
        formatted_data = {"tokens": [], "ner_tags": []}
        for sentence in data:
            tokens = [token_data[0].lower() for token_data in sentence]
            # 2 is 'O' tag in NER mountain training
            # since this dataset almost doesn't contain mountain entities
            # we make all tokens 'O'
            ner_tags = [2] * len(sentence)
            formatted_data["tokens"].append(tokens)
            formatted_data["ner_tags"].append(ner_tags)

        return formatted_data

    @staticmethod
    def formate_raw_data_txt(raw_data: list, label_map: dict) -> dict:
        """
    Create {"tokens": [], "ner_tags": []} based on the conll2003 like raw data

    :param raw_data: conll2003 like raw data
    :param label_map: # {'ENTITY-TAG': 0 .., 'O': n-1}
    :return: {"tokens": [], "ner_tags": []}
    """
        formatted_custom_data = {"tokens": [], "ner_tags": []}
        for sentence in raw_data:
            tokens = [token_data[0].lower() for token_data in sentence]
            ner_tags = [label_map[token_data[1]] for token_data in sentence]
            formatted_custom_data["tokens"].append(tokens)
            formatted_custom_data["ner_tags"].append(ner_tags)

        return formatted_custom_data

    def get_conll_formatted_data_for_adding(self) -> dict:
        train_data = read_conll_like_file(self.conll_train)
        formatted_data = self.formate_conll_data_for_adding(train_data)

        return formatted_data

    def get_split_formatted_data(self, improve_with_conll: int = 2, train_proportion: float = 0.7):
        """
    Create train_set_dict, val_set_dict, test_set_dict in {"tokens": [], "ner_tags": []} format each

    :param improve_with_conll: multiplicator for calculation how many conll data should be added
    based on the size of original data (0 - no conll data added, 2 - 2x of original data size added)
    :param train_proportion: proportion of how big train set should be (0.7 - 70% of data is train set)
    :return: 3 dicts train_set_dict, val_set_dict, test_set_dict
    """

        raw_data = read_conll_like_file(self.dataset_mountain_path)

        # for mountain dataset {'B-MON': 0, 'I-MON': 1, 'O': 2}
        self.label_list = sorted(list(set([token_data[1] for sentence in raw_data for token_data in sentence])))
        self.label_map = {label: i for i, label in enumerate(self.label_list)}

        formatted_data = self.formate_raw_data_txt(raw_data, self.label_map)
        data_size = len(formatted_data['tokens'])

        if improve_with_conll > 0:
            formatted_data = self.add_conll_data(formatted_data, improve_with_conll)

        val_part = int((1 - train_proportion) * data_size)
        train_set_dict, val_set_dict = self.create_test_set(formatted_data, val_part)

        # FIXME: hardcode proportion
        # for simplicity assume that validation/test set have proportion 60/40
        test_part = int(val_part * 0.4)
        val_set_dict, test_set_dict = self.create_test_set(val_set_dict, test_part)

        return train_set_dict, val_set_dict, test_set_dict

    def add_conll_data(self, formatted_data: dict, improve_with_conll: int) -> dict:
        """
    Add part of conll2003 data set to the original data for diversity

    :param formatted_data: {"tokens": [], "ner_tags": []}
    :param improve_with_conll: multiplicator for calculation how many conll data should be added
    based on the size of original data (0 - no conll data added, 2 - 2x of original data size added)
    :return: {"tokens": [], "ner_tags": []}
    """

        data_size = len(formatted_data['tokens'])

        conll_formatted_data = self.get_conll_formatted_data_for_adding()
        conll_tokens = conll_formatted_data["tokens"]
        conll_tags = conll_formatted_data["ner_tags"]

        print(f"dataset size after before conll: {data_size}")

        # FIXME: add random with reproducibility
        formatted_data["tokens"].extend(conll_tokens[100:data_size * improve_with_conll])
        formatted_data["ner_tags"].extend(conll_tags[100:data_size * improve_with_conll])

        print(f"dataset size after adding conll: {len(formatted_data['tokens'])}")

        return formatted_data

    @staticmethod
    def create_test_set(formatted_data: dict, part: int) -> tuple[dict, dict]:
        len_data = len(formatted_data["tokens"])

        # FIXME: doesn't have reproducibility
        random_position = random.sample(range(0, len_data), part)

        formatted_data_val = {"tokens": [], "ner_tags": []}
        formatted_data_train = {"tokens": [], "ner_tags": []}

        formatted_data_val["tokens"] = [formatted_data["tokens"][i] for i in random_position]
        formatted_data_val["ner_tags"] = [formatted_data["ner_tags"][i] for i in random_position]

        formatted_data_train["tokens"] = [formatted_data["tokens"][i] for i in range(len_data) if
                                          i not in random_position]
        formatted_data_train["ner_tags"] = [formatted_data["ner_tags"][i] for i in range(len_data) if
                                            i not in random_position]

        return formatted_data_train, formatted_data_val
