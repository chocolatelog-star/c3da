# Restaurant C3DA Run Analysis

This report analyzes the completed run:

```text
dataset = restaurant
classifier = roberta
generator = t5 + lora
seed = 1000
generator epochs = 100
classifier epochs = 15
classifier batch size = 8
augmentation = enabled
contrastive learning = enabled
```

## 1. Final Result

The final logged best result is:

```text
max_test_acc_overall = 0.868632707774799
max_f1_overall       = 0.8193910518339607
```

So this single-seed run obtains:

```text
Accuracy = 86.86%
Macro-F1 = 81.94%
```

The final classification report is:

```text
positive F1 = 0.9252
negative F1 = 0.8039
neutral  F1 = 0.7185
```

The model performs best on positive samples and weakest on neutral samples.

## 2. Main Confusion Pattern

The confusion matrix is:

```text
[[674  29  24]
 [ 13 164  19]
 [ 43  19 134]]
```

Rows are gold labels and columns are predicted labels.

Label mapping in this project:

```text
0 = positive
1 = negative
2 = neutral
```

Main errors:

```text
neutral -> positive: 43 cases
positive -> negative: 29 cases
positive -> neutral: 24 cases
negative -> neutral: 19 cases
neutral -> negative: 19 cases
```

The largest error source is:

```text
neutral samples are often predicted as positive.
```

This indicates that the classifier is biased toward positive expressions, especially when the sentence has weak sentiment or ambiguous sentiment.

## 3. Class Imbalance

The test distribution is:

```text
positive: 727
negative: 196
neutral:  196
```

Positive samples account for about 65% of the test set.

This affects the model in two ways:

1. Accuracy looks high because the majority class is positive.
2. Macro-F1 exposes the weakness on minority classes, especially neutral.

This is why:

```text
Accuracy = 86.86%
Macro-F1 = 81.94%
```

Accuracy is much higher than neutral F1.

## 4. Generated Data Quality

The generated augmentation file is:

```text
dataset/Restaurants_corenlp/generate-t5-lora-100.json
```

Statistics:

```text
groups           = 3608
total generated  = 14432
empty generated  = 0
unique generated = 1914
duplicate extra  = 12518
```

This is the most important weakness discovered in this run.

Although 14432 generated sentences were produced, only 1914 are unique. The generator repeats many generic restaurant sentences.

Most repeated examples:

```text
1668 times: The food is delicious and I highly recommend it.
1577 times: The food is delicious and the service is prompt and professional.
903 times:  The food is very good too..
456 times:  The food is delicious and the prices are reasonable.
277 times:  The food is delicious and the staff is very attentive.
```

This shows that the generator collapses toward high-frequency positive restaurant templates.

## 5. Main Loss Source

The main performance loss is not that the classifier cannot fit the training data.

Training logs show that late-stage training accuracy reaches around:

```text
train acc ≈ 0.99 - 1.00
```

But test F1 still fluctuates and does not keep improving.

This means the main loss is:

```text
generalization loss
```

not:

```text
training optimization failure
```

In other words, the model learns the training set very well, but some learned patterns do not transfer cleanly to the test set.

## 6. Augmentation Loss Behavior

The logs show:

```text
VanillaLoss and AugLoss are often similar in early training.
```

This means the augmented samples are being used actively and are not ignored.

However, because many generated samples are duplicated and generic, AugLoss may reinforce shallow patterns such as:

```text
food + delicious -> positive
service + professional -> positive
```

This helps positive classification, but may hurt neutral and subtle sentiment cases.

## 7. Contrastive Loss Issue

Although the command enabled contrastive learning:

```text
--withCL
```

the logged `CLLoss` is almost always:

```text
CLLoss: 0.0000
```

This suggests that the contrastive objective is contributing very little in this run.

Possible reasons:

1. `k = 1`, so only one augmented candidate is selected per sample.
2. The generated positive and negative representations may already satisfy the margin.
3. The contrastive setup is not strong enough under the current generated data distribution.

This is important because C3DA's theoretical strength is contrastive robustness, but in this run the contrastive signal is weak.

## 8. Specific Weaknesses

### 8.1 Neutral Classification Is Weak

Neutral F1 is:

```text
0.7185
```

Neutral recall is:

```text
0.6837
```

The model misses many neutral samples, often predicting them as positive.

Likely causes:

- Positive class is much larger.
- Generated data contains many positive generic sentences.
- Neutral expressions are semantically ambiguous.
- T5 generation may not preserve neutral polarity well.

### 8.2 Generated Data Is Repetitive

The generator produces many duplicate or near-duplicate samples.

This reduces the benefit of data augmentation because the effective data size is far smaller than the raw generated count.

The real augmentation diversity is closer to:

```text
1914 unique sentences
```

not:

```text
14432 generated sentences
```

### 8.3 Aspect Diversity Is Limited

Repeated generated sentences are heavily centered on:

```text
food
service
staff
prices
atmosphere
```

This means long-tail aspects may not be sufficiently covered.

Examples of likely under-covered aspects:

```text
wine
seats
menu
quantity
design
specific dishes
```

This limits multi-aspect robustness.

### 8.4 Polarity Preservation Is Imperfect

Some generated examples are fluent but may drift in aspect or sentiment.

For example, during generation/fine-tuning, some case studies showed source sentences being rewritten into generic positive food/service sentences.

This can inject noisy augmented labels.

### 8.5 Overfitting Appears Late in Training

Late logs show very high training accuracy, but test scores fluctuate.

This indicates that 15 epochs may be longer than necessary for this run, or that stronger early stopping should be used.

The saved best model is selected by best observed test accuracy/F1, but the training process itself keeps fitting the training set strongly.

## 9. Recommended Improvements

### 9.1 Add Duplicate Filtering

Before training, remove exact duplicate generated sentences.

This alone can prevent repeated generic samples from dominating training.

### 9.2 Add Semantic Deduplication

Exact deduplication is not enough because many generated sentences are near duplicates.

Use sentence embeddings to remove near-duplicates:

```text
cosine similarity > 0.95 -> keep one
```

### 9.3 Add NLI Filtering

Borrow from RSDA:

```text
premise = source/original sentence
hypothesis = generated sentence
```

Remove samples with high contradiction probability:

```text
P(contradiction) >= 0.5
```

This can reduce polarity drift and semantic mismatch.

### 9.4 Balance Generated Polarities

The generated data is biased toward positive templates.

Force generation or sampling to balance:

```text
positive
negative
neutral
```

Especially increase high-quality neutral and negative augmentations.

### 9.5 Use Domain/Aspect Fragment Control

Borrow from DAEGCN:

Extract domain-specific aspect/opinion fragments using frequency-ratio scoring.

Then guide generation with selected fragments so the generator does not collapse into only food/service templates.

### 9.6 Strengthen Contrastive Learning

Because CLLoss is nearly zero, consider:

```text
increase k
tune margin
use harder negatives
lower cl_loss_fac only after CLLoss becomes meaningful
```

For example:

```text
--k 2
--margin 0.3 or 0.7
```

But increasing `k` will use more VRAM.

### 9.7 Use Early Stopping

Instead of always training 15 epochs, stop if validation/test F1 does not improve for several evaluations.

This can reduce overfitting and save time.

## 10. Summary

This run is successful as a single-seed Restaurant reproduction:

```text
Accuracy = 86.86%
Macro-F1 = 81.94%
```

However, the main weaknesses are:

```text
1. Neutral class performance is weak.
2. Generated data is highly repetitive.
3. Generated samples are biased toward positive restaurant templates.
4. Long-tail aspects are under-covered.
5. Contrastive loss contributes very little in this run.
6. Training fits the training set strongly, but test F1 fluctuates.
```

The main loss is therefore:

```text
generalization loss caused by class imbalance, low-diversity augmentation, and noisy or generic generated samples.
```

The most promising next improvement is:

```text
deduplicate generated data + NLI filtering + domain-specific fragment-guided generation.
```

