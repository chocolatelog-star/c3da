@echo off
J:
cd /d J:\nlp\CD-C3DA\.worktrees\reproduce-best-0332aee
echo started %date% %time% > runs\best_0332aee_full_rerun.status.txt
J:\conda\envs\c3da\python.exe run_bgca_aste_stage1_pairs.py --output_root runs\bgca_aste_stage1_best_0332aee_full_rerun_v1 --pairs rest16:laptop14 --extractor_model_path J:\nlp\models\t5-base-py --generator_model_path J:\nlp\models\t5-base-py --generator_prompt_style label_to_text --augment_prompt_style masked_mutual --domain_prefix_style text --extractor_epochs 25 --generator_epochs 8 --final_epochs 5 --complete_multi_extra_weight 0.25 --final_pseudo_weight 0.65 --final_augment_weight 0.20 --lambda_sentiment_contrastive 0.01 --sentiment_contrastive_source_only --sentiment_contrastive_class_balanced --learning_rate 0.0003 --eval_batch_size 2 --cuda 0 --seed 1000 > runs\best_0332aee_full_rerun.out.log 2> runs\best_0332aee_full_rerun.err.log
echo exited %errorlevel% %date% %time% >> runs\best_0332aee_full_rerun.status.txt
