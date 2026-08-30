import torch

from t5_absa_train import DataCollatorForSeq2SeqWithPairing, JsonlSeq2SeqDataset


class TinyTokenizer:
    pad_token_id = 0

    def __call__(self, text=None, text_target=None, max_length=None, truncation=True):
        value = text_target if text_target is not None else text
        return {"input_ids": [1] * min(len(str(value)), max_length or len(str(value)))}

    def encode(self, text, add_special_tokens=False):
        return [1] * len(str(text))


class FakeGraphCache:
    def get(self, row):
        if str(row["id"]) == "1":
            return {
                "word_to_subword": [[0, 1], [2]],
                "word_mask": [1, 1],
                "edge_src": [0, 1],
                "edge_dst": [0, 1],
                "relation_id": [0, 1],
                "dependency_relation_id": [0, 1],
                "pos_pair_id": [0, 1],
                "edge_mask": [1, 1],
            }
        return {
            "word_to_subword": [[0, 1], [2], [3, 4]],
            "word_mask": [1, 1, 1],
            "edge_src": [0, 1, 2],
            "edge_dst": [0, 1, 2],
            "relation_id": [0, 1, 2],
            "dependency_relation_id": [0, 1, 2],
            "pos_pair_id": [0, 1, 2],
            "edge_mask": [1, 1, 1],
        }

    def get_parser_tokens(self, row):
        text = row["text"]
        words = text.split()
        tokens = []
        cursor = 0
        for index, word in enumerate(words):
            start = text.index(word, cursor)
            end = start + len(word)
            tokens.append({"index": index, "text": word, "start": start, "end": end})
            cursor = end
        return tokens


class BaseCollator:
    def __call__(self, features):
        max_input = max(len(feature["input_ids"]) for feature in features)
        max_label = max(len(feature["labels"]) for feature in features)
        return {
            "input_ids": torch.tensor(
                [feature["input_ids"] + [0] * (max_input - len(feature["input_ids"])) for feature in features],
                dtype=torch.long,
            ),
            "labels": torch.tensor(
                [feature["labels"] + [0] * (max_label - len(feature["labels"])) for feature in features],
                dtype=torch.long,
            ),
        }


def test_graph_dataset_and_collator_pad_without_cross_sample_edges():
    rows = [
        {"id": 1, "input": "abcd", "target": "x", "text": "abcd"},
        {"id": 2, "input": "abcde", "target": "x", "text": "abcde"},
    ]
    dataset = JsonlSeq2SeqDataset(
        rows,
        TinyTokenizer(),
        max_source_length=8,
        max_target_length=8,
        source_weight=1.0,
        pseudo_weight=0.5,
        augment_weight=0.2,
        graph_cache=FakeGraphCache(),
    )
    features = [dataset[0], dataset[1]]
    batch = DataCollatorForSeq2SeqWithPairing(BaseCollator())(features)

    assert tuple(batch["graph_word_to_subword"].shape) == (2, 3, 2)
    assert tuple(batch["graph_edge_src"].shape) == (2, 3)
    assert batch["graph_word_mask"].tolist() == [[True, True, False], [True, True, True]]
    assert batch["graph_edge_mask"].tolist() == [[True, True, False], [True, True, True]]
    assert batch["graph_edge_src"][0, 2].item() == 0


def test_element_aware_dataset_supervises_source_only_and_collator_pads_nodes():
    rows = [
        {
            "id": 1,
            "input": "food good",
            "text": "food good",
            "target": "<pos> food <opinion> good",
        },
        {
            "id": 2,
            "input": "service slow",
            "text": "service slow",
            "target": "<neg> service <opinion> slow",
            "augmentation": "target_unlabeled",
        },
    ]
    dataset = JsonlSeq2SeqDataset(
        rows,
        TinyTokenizer(),
        16,
        16,
        1.0,
        0.5,
        0.2,
        graph_cache=FakeGraphCache(),
        element_aware_enabled=True,
    )
    batch = DataCollatorForSeq2SeqWithPairing(BaseCollator())([dataset[0], dataset[1]])

    assert batch["element_source_row"].tolist() == [True, False]
    assert batch["element_node_labels"][0, :2].tolist() == [1, 1]
    assert batch["element_node_loss_mask"][0, :2].tolist() == [True, True]
    assert not batch["element_node_loss_mask"][1].any()
    assert batch["element_triplet_count"].tolist() == [1, 0]
