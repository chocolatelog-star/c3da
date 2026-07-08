import argparse
import os

from huggingface_hub import snapshot_download


MODELS = {
    "t5": "mrm8488/t5-base-finetuned-common_gen",
    "roberta": "roberta-base",
    "bert": "bert-base-uncased",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_root", default=r"J:\nlp\models")
    parser.add_argument("--models", default="t5,roberta,bert")
    args = parser.parse_args()

    os.makedirs(args.model_root, exist_ok=True)
    for key in [item.strip() for item in args.models.split(",") if item.strip()]:
        repo_id = MODELS[key]
        local_dir = os.path.join(args.model_root, repo_id.replace("/", "-"))
        print("Downloading {} -> {}".format(repo_id, local_dir))
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            resume_download=True,
        )


if __name__ == "__main__":
    main()
