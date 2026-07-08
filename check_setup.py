import os

import torch
import transformers
from transformers import AutoModel, RobertaTokenizer, T5ForConditionalGeneration, T5Tokenizer


MODEL_ROOT = r"J:\nlp\models"
T5_PATH = os.path.join(MODEL_ROOT, "mrm8488-t5-base-finetuned-common_gen")
ROBERTA_PATH = os.path.join(MODEL_ROOT, "roberta-base")
BERT_PATH = os.path.join(MODEL_ROOT, "bert-base-uncased")


def check_dir(path):
    if not os.path.isdir(path):
        raise FileNotFoundError("Missing model directory: {}".format(path))
    print("found {}".format(path))


def main():
    print("torch:", torch.__version__)
    print("transformers:", transformers.__version__)
    print("cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("gpu:", torch.cuda.get_device_name(0))

    check_dir(T5_PATH)
    check_dir(ROBERTA_PATH)
    check_dir(BERT_PATH)

    T5Tokenizer.from_pretrained(T5_PATH)
    T5ForConditionalGeneration.from_pretrained(T5_PATH)
    RobertaTokenizer.from_pretrained(ROBERTA_PATH)
    AutoModel.from_pretrained(ROBERTA_PATH)
    print("setup ok")


if __name__ == "__main__":
    main()
