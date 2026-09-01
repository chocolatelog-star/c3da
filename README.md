# C3DA

## 可控实验入口

- `run_upstream_batch_matrix.py`：只比较上游 batch，并汇总 source-dev F1 与伪标签产物。
- `run_fixed_upstream_downstream_batch_matrix.py`：冻结同一训练数据和全部上游身份，只比较最终训练 batch。
- `batch_gradient_parameter_audit.py`：比较 `1x16/4x4/8x2/16x1` 的单步梯度、参数更新和真实有效样本归一化。
- `run_graph_control_ab.py`：在同一配置下完成 Graph OFF/ON 公平 A/B 并输出最终 F1 差值。

历史最佳训练继续使用 `t5_absa_train.py`；图实验使用隔离的 `t5_absa_train_graph.py`。

Code and datasets of our paper “A Contrastive Cross-Channel Data Augmentation Framework for Aspect-based Sentiment Analysis”



## Requirements

- torch==1.4.0
- scikit-learn==0.23.2
- transformers==3.2.0
- cython==0.29.13
- nltk==3.5

To install requirements, run `pip install -r requirements.txt`.



## Generating

To generate data items, run:

`python C3DA/generate.py`



## Training

To train the C3DA model, run:

`sh C3DA/run.sh`

and `C3DA/start.sh`, `C3DA/start1.sh` is used to adjust our hyper-parameters.



## Logs

Logs are saved under `C3DA/C3DA/log`



## Credits

The code and datasets in this repository are based on [ABSA-PyTorch](https://github.com/songyouwei/ABSA-PyTorch) and [CDT_ABSA](https://github.com/Guangzidetiaoyue/CDT_ABSA).


## Cite

```
@inproceedings{wang2022a,
  author    = {Bing Wang and Liang Ding and Qihuang Zhong and Ximing Li and Dacheng Tao},
  title     = {A Contrastive Cross-Channel Data Augmentation Framework for Aspect-Based Sentiment Analysis},
  booktitle = {Proceedings of the 29th International Conference on Computational Linguistics, {COLING} 2022, Gyeongju, Republic of Korea, October 12-17,
               2022},
  pages     = {6691--6704},
  publisher = {International Committee on Computational Linguistics},
  year      = {2022},
  url       = {https://aclanthology.org/2022.coling-1.581},
  timestamp = {Thu, 13 Oct 2022 17:29:38 +0200},
  biburl    = {https://dblp.org/rec/conf/coling/Wang0ZLT22.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```


