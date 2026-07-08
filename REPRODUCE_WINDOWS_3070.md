# Windows RTX 3070 Reproduction

This project is the C3DA paper code. The original `requirements.txt` pins very old CUDA/PyTorch packages, so use this Windows + RTX 3070 setup instead. All commands below are one-line CMD commands.

## 1. Create Environment

`conda create -n c3da python=3.10 -y`

`conda activate c3da`

## 2. Install GPU PyTorch

For RTX 3070, install a CUDA 12.1 PyTorch build:

`python -m pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121`

Then install the remaining packages:

`python -m pip install transformers==4.39.3 huggingface-hub==0.36.0 sentencepiece==0.2.0 scikit-learn==1.4.2 numpy==1.26.4 tqdm==4.66.2 protobuf==4.25.8`

Check CUDA:

`python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CUDA not available')"`

## 3. Download Models to J:\nlp\models

From `J:\nlp\C3DA-main`, run:

`python download_models.py --model_root J:\nlp\models --models t5,roberta,bert`

Expected directories:

`J:\nlp\models\mrm8488-t5-base-finetuned-common_gen`

`J:\nlp\models\roberta-base`

`J:\nlp\models\bert-base-uncased`

Verify the whole setup:

`python check_setup.py`

## 4. Generate Augmented Data

Restaurant, LoRA generator, RTX 3070-safe batch size:

`python generate.py --dataset restaurant --prompt_name lora --num_epoch 100 --batch_size 4 --num_workers 0 --cuda 0 --model_root J:\nlp\models`

The output should be:

`J:\nlp\C3DA-main\dataset\Restaurants_corenlp\generate-t5-lora-100.json`

## 5. Train C3DA

Start with one seed:

`python train.py --dataset restaurant --model_name roberta --generate_model t5 --prompt_name lora --ft_epoch 100 --withAugment --withCL --aug_loss_fac 1.0 --margin 0.5 --cl_loss_fac 2.0 --k 1 --num_epoch 15 --batch_size 8 --cuda 0 --model_root J:\nlp\models --seed 1000`

For paper-style five-seed runs, repeat the same command with `--seed 2000`, `--seed 3000`, `--seed 4000`, and `--seed 5000`.

## 6. Other Datasets

Laptop:

`python generate.py --dataset laptop --prompt_name lora --num_epoch 100 --batch_size 4 --num_workers 0 --cuda 0 --model_root J:\nlp\models`

`python train.py --dataset laptop --model_name roberta --generate_model t5 --prompt_name lora --ft_epoch 100 --withAugment --withCL --aug_loss_fac 1.0 --margin 0.5 --cl_loss_fac 2.0 --k 1 --num_epoch 15 --batch_size 8 --cuda 0 --model_root J:\nlp\models --seed 1000`

Twitter:

`python generate.py --dataset twitter --prompt_name lora --num_epoch 100 --batch_size 4 --num_workers 0 --cuda 0 --model_root J:\nlp\models`

`python train.py --dataset twitter --model_name roberta --generate_model t5 --prompt_name lora --ft_epoch 100 --withAugment --withCL --aug_loss_fac 1.0 --margin 0.5 --cl_loss_fac 2.0 --k 1 --num_epoch 15 --batch_size 8 --cuda 0 --model_root J:\nlp\models --seed 1000`

## Paper Hyperparameters

The paper uses T5 with Adafactor, 100 generator fine-tuning epochs, batch size 16, LoRA rank 8 and dropout 0. Prediction models use Adam/AdamW, learning rate `2e-5`, dropout `0.3`, 15 epochs, batch size 16, and five random seeds. Since RTX 3070 has 8GB VRAM, use batch size 4 for generation and 8 for training first; raise them only if memory allows.
