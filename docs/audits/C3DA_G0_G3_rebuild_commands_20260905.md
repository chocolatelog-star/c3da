# G0/G3 完整下游重建命令

服务器实例当前必须先出现 GPU（显卡）后再执行。两组使用现有 extractor checkpoint（提取器检查点），不重新训练 extractor。

## G0

```bash
cd /root/CD-C3DA-chat-convergence && export OMP_NUM_THREADS=8 && nohup /root/miniconda3/envs/c3da/bin/python3.10 run_bgca_aste_stage1_pairs.py --pairs laptop14:rest15 --output_root /root/autodl-tmp/CD-C3DA-runs/rebuild_G0_20260905 --reuse_upstream_run_dir /root/autodl-tmp/CD-C3DA-runs/chat_l14_r15_G0_16x2_20260904_run2_full_adapter --extractor_model_path /root/CD-C3DA-pre-git-20260830/models/t5-base-py --generator_model_path /root/CD-C3DA-pre-git-20260830/models/t5-base-py --generator_prompt_style label_to_text --augment_prompt_style masked_mutual --nli_model_path /root/CD-C3DA-pre-git-20260830/models/nli-deberta-v3-base-mnli-fever-anli --sentiment_vector_model_path /root/CD-C3DA-pre-git-20260830/models/t5-base-py --glove_path /root/CD-C3DA-pre-git-20260830/models/glove/glove.6B.300d.txt --complete_multi_extra_weight 0.25 --final_pseudo_weight 0.75 --final_augment_weight 0.2 --lambda_sentiment_contrastive 0.01 --sentiment_contrastive_source_only --sentiment_contrastive_class_balanced --train_batch_size 16 --gradient_accumulation_steps 2 --eval_batch_size 16 --learning_rate 0.0003 --minimal_outputs --final_lambda_domain_adv 0.03 --cuda 0 --seed 1000 > /root/autodl-tmp/CD-C3DA-runs/rebuild_G0_20260905.log 2>&1 &
```

## G3

```bash
cd /root/CD-C3DA-chat-convergence && export OMP_NUM_THREADS=8 && nohup /root/miniconda3/envs/c3da/bin/python3.10 run_bgca_aste_stage1_pairs.py --pairs laptop14:rest15 --output_root /root/autodl-tmp/CD-C3DA-runs/rebuild_G3_20260905 --reuse_upstream_run_dir /root/autodl-tmp/CD-C3DA-runs/chat_l14_r15_G3_16x2_20260904_run2_full_adapter --extractor_model_path /root/CD-C3DA-pre-git-20260830/models/t5-base-py --generator_model_path /root/CD-C3DA-pre-git-20260830/models/t5-base-py --generator_prompt_style label_to_text --augment_prompt_style masked_mutual --nli_model_path /root/CD-C3DA-pre-git-20260830/models/nli-deberta-v3-base-mnli-fever-anli --sentiment_vector_model_path /root/CD-C3DA-pre-git-20260830/models/t5-base-py --glove_path /root/CD-C3DA-pre-git-20260830/models/glove/glove.6B.300d.txt --complete_multi_extra_weight 0.25 --final_pseudo_weight 0.75 --final_augment_weight 0.2 --lambda_sentiment_contrastive 0.01 --sentiment_contrastive_source_only --sentiment_contrastive_class_balanced --train_batch_size 16 --gradient_accumulation_steps 2 --eval_batch_size 16 --learning_rate 0.0003 --minimal_outputs --final_lambda_domain_adv 0.03 --cuda 0 --seed 1000 > /root/autodl-tmp/CD-C3DA-runs/rebuild_G3_20260905.log 2>&1 &
```

监控：

```bash
nvidia-smi
tail -f /root/autodl-tmp/CD-C3DA-runs/rebuild_G0_20260905.log
tail -f /root/autodl-tmp/CD-C3DA-runs/rebuild_G3_20260905.log
```
