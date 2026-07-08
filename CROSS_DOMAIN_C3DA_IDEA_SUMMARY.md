# Cross-domain C3DA Improvement Notes

This document summarizes our design discussion about improving C3DA for cross-domain ABSA by borrowing ideas from two papers:

1. RSDA: "Refining and Synthesis: A Simple yet Effective Data Augmentation Framework for Cross-Domain Aspect-based Sentiment Analysis"
2. DAEGCN: "Data augmentation framework with enhanced graph convolutional network for cross-domain aspect-based sentiment analysis"

The purpose of this document is to preserve the full idea so that future long conversations do not lose the design context.

## 1. Original C3DA Role Division

Original C3DA is not just a T5 classifier.

Its pipeline is:

```text
Original ABSA data
        |
        v
T5 + LoRA generator
        |
        v
Generated augmented samples
        |
        v
BERT/RoBERTa ABSA classifier
        |
        v
Contrastive learning + classification
```

In C3DA:

- T5 is the generator.
- LoRA is used to fine-tune T5 efficiently.
- BERT/RoBERTa is the final ABSA sentiment classifier.
- Contrastive learning improves robustness against multi-aspect sentiment interference.

The current project mainly handles in-domain ABSA, such as Restaurant -> Restaurant.

Our goal is to extend it into cross-domain ABSA, such as:

```text
Restaurant -> Laptop
Laptop -> Restaurant
Restaurant -> Twitter
Twitter -> Restaurant
```

## 2. Target Improvement Direction

We want to build a cross-domain version of C3DA:

```text
Cross-domain C3DA / CD-C3DA / RSDA-C3DA
```

Core idea:

```text
Use source-domain labeled data and target-domain unlabeled data to generate high-quality target-domain pseudo-labeled data, then train a robust ABSA classifier with contrastive learning.
```

High-level pipeline:

```text
Source labeled data Ds
Target unlabeled data Ut
        |
        v
Target-domain aspect/opinion fragment extraction
        |
        v
Pseudo-label generation
        |
        v
T5/LoRA cross-domain data generation
        |
        v
NLI quality filtering
        |
        v
Optional diversity enhancement
        |
        v
RoBERTa/BERT classifier + C3DA contrastive learning
```

## 3. Borrowed Idea From RSDA

RSDA focuses on cross-domain ABSA data augmentation.

Its key problem statement:

- Target-domain unlabeled data are often pseudo-labeled by a source-trained model.
- Pseudo labels can be noisy.
- Generated data based on noisy pseudo labels can propagate errors.
- Generated samples can also be monotonous and lack diversity.

RSDA addresses this with two major steps:

```text
1. Data generation and quality control
2. Data diversity augmentation
```

### 3.1 Pseudo-label Driven Generation

RSDA uses a source-trained extraction model to obtain pseudo labels from target-domain unlabeled text, then uses a generation model to generate target-domain labeled samples.

For our C3DA extension, the pseudo label should be adapted to the current task.

Current C3DA is aspect-based sentiment classification, so a pseudo label can be:

```text
(sentence, aspect, polarity)
```

Example:

```text
Target sentence:
The screen is bright but the battery dies quickly.

Pseudo labels:
(screen, positive)
(battery, negative)
```

These pseudo labels can be converted into generation conditions:

```text
aspect: screen sentiment: positive
aspect: battery sentiment: negative
```

or compact labels:

```text
<pos> screen
<neg> battery
```

### 3.2 NLI Filtering

RSDA uses an NLI model as a quality filter.

Original RSDA setting:

```text
premise = original target-domain text t
hypothesis = generated text t'
```

The NLI model predicts:

```text
entailment / neutral / contradiction
```

If contradiction is high, the generated sample should be removed.

For our framework, this is useful because cross-domain generation can drift away from the original meaning or generate polarity-inconsistent samples.

Recommended first-version rule:

```text
Keep sample if P(contradiction) < 0.5
```

Stricter version:

```text
Keep sample if P(entailment) > 0.5 and P(contradiction) < 0.2
```

But because data augmentation does not always require strict entailment, the safer first version is:

```text
Only filter high-contradiction samples.
```

### 3.3 NLI Filter and C3DA Entropy Filter

Original C3DA has entropy-based filtering.

We can combine both:

```text
1. NLI filter: remove semantic contradiction or obvious generation drift.
2. Entropy filter: remove samples where the classifier is uncertain.
```

Recommended order:

```text
Generated samples -> NLI filter -> entropy filter -> training
```

NLI works at the text semantic consistency level.
Entropy works at the classifier confidence/stability level.

### 3.4 Label Composition

RSDA also uses label composition to increase diversity and information density.

For our cross-domain ABSA setting, label composition is especially useful for multi-aspect sentiment:

```text
(screen, positive) + (battery, negative)
```

Then T5 generates:

```text
The screen is excellent, but the battery life is disappointing.
```

This directly supports the original C3DA motivation: robustness under multi-aspect, multi-polarity sentences.

## 4. Borrowed Idea From DAEGCN

DAEGCN is another cross-domain ABSA data augmentation framework.

Its key ideas:

- Domain-specific segment aware attention.
- Linguistic feature knowledge.
- Domain adversarial learning.
- GCN for target-domain pseudo-label annotation.
- Masking noun/adjective fragments and generating target-domain labeled data.

### 4.1 Domain-specific Segment Awareness

This is the most immediately useful part for our framework.

DAEGCN observes:

```text
Domain differences are mainly reflected in nouns and adjectives.
```

Nouns often represent domain-specific entities/aspects:

```text
Restaurant: food, service, staff, sushi
Laptop: battery, screen, keyboard, processor
```

Adjectives often represent opinion/sentiment:

```text
good, bad, fresh, slow, noisy, bright
```

Instead of simply extracting high-frequency words from the target domain, we should extract domain-specific fragments.

Simplified scoring:

```text
score(z) = freq_target(z) / (freq_source(z) + epsilon)
```

Where `z` is an n-gram fragment.

Keep a fragment if:

```text
score(z) >= threshold
and z contains a noun / noun phrase / adjective phrase
```

This is better than pure frequency extraction, because pure frequency may select uninformative words such as:

```text
the, this, very, good
```

Domain-specific fragment extraction is more likely to select:

```text
battery life, screen, keyboard, fried rice, wait staff
```

### 4.2 How To Use Domain-specific Fragments

We should use this method to build a target-domain aspect/opinion vocabulary.

Pipeline:

```text
Source corpus + target corpus
        |
        v
n-gram extraction
        |
        v
POS filtering / noun phrase filtering
        |
        v
frequency-ratio scoring
        |
        v
target-domain specific aspect/opinion fragments
        |
        v
T5 generation conditions
```

Example:

```text
Source domain: Restaurant
Target domain: Laptop

Extracted target fragments:
battery life, screen, keyboard, fan noise

Generation condition:
aspect: battery life sentiment: negative
```

### 4.3 Whether To Use GCN

DAEGCN uses GCN to learn better representations and annotate target-domain pseudo labels.

This can be useful, but should not be the first implementation step.

Reason:

- Current C3DA codebase is lightweight.
- Full DAEGCN requires graph construction, dependency/linguistic features, domain adversarial training, and GCN layers.
- Adding all of this immediately would greatly increase engineering complexity.

Recommended phased plan:

```text
Version 1:
Use domain-specific fragment extraction only.

Version 2:
Use source-trained RoBERTa to pseudo-label target fragments/sentences.

Version 3:
Introduce GCN-enhanced pseudo-labeler with domain adversarial learning.
```

So the answer is:

```text
Use DAEGCN's domain-specific segment awareness immediately.
Use DAEGCN's GCN as a later-stage pseudo-labeler reference, not as the first implementation.
```

## 5. Proposed Final Framework

The improved cross-domain framework should combine:

- C3DA: T5/LoRA generation, contrastive learning, entropy filtering.
- RSDA: pseudo-label generation, NLI filtering, label composition/paraphrase diversity.
- DAEGCN: domain-specific fragment extraction, optional GCN-based pseudo-labeling.

Recommended full framework:

```text
Source labeled data Ds
Target unlabeled data Ut
        |
        v
Domain-specific fragment extraction
        |
        v
Target aspect/opinion candidate vocabulary
        |
        v
Pseudo-label generation
        |
        v
T5/LoRA cross-domain generation
        |
        v
NLI contradiction filtering
        |
        v
Optional label composition for multi-aspect samples
        |
        v
RoBERTa/BERT classifier training
        |
        v
C3DA contrastive learning + entropy filtering
```

## 6. Suggested Implementation Plan

### Stage 1: Minimal Cross-domain Version

Implement:

```text
extract_domain_fragments.py
```

Input:

```text
source dataset
target dataset
```

Output:

```text
target_domain_fragments.json
```

This file contains target-domain aspect/opinion candidates.

Then modify or extend generation:

```text
cross_generate.py
```

Input:

```text
source dataset
target fragment vocabulary
sentiment polarity
```

Output:

```text
generated target-domain pseudo-labeled samples
```

### Stage 2: NLI Filtering

Implement:

```text
nli_filter.py
```

Input:

```text
original target text
generated text
pseudo label
```

Output:

```text
filtered generated target-domain samples
```

Recommended model:

```text
MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli
```

Local path suggestion:

```text
J:\nlp\models\nli-deberta-v3-base-mnli-fever-anli
```

### Stage 3: Label Composition

Add target-domain label composition:

```text
(aspect_1, polarity_1) + (aspect_2, polarity_2)
```

Example:

```text
(screen, positive) + (battery, negative)
```

This should generate multi-aspect, multi-polarity samples.

This step is highly aligned with the original C3DA problem setting.

### Stage 4: Optional GCN Pseudo-labeler

If the above works, then consider:

```text
gcn_pseudo_labeler.py
```

It can use:

- BERT/RoBERTa token representations.
- Dependency graph or co-occurrence graph.
- Domain-specific fragment features.
- Domain adversarial loss.

But this should be treated as a later extension.

## 7. Possible Paper Contributions

The improved method can claim:

```text
1. Domain-specific fragment aware target aspect extraction.
2. Pseudo-label driven cross-domain C3DA generation.
3. NLI-based quality control for generated target-domain samples.
4. Label composition for multi-aspect, multi-polarity target samples.
5. Contrastive training for cross-domain ABSA robustness.
```

Potential method name:

```text
NLI-CD-C3DA
Fragment-aware CD-C3DA
RSDA-C3DA
DA-C3DA
```

## 8. Experimental Design

Use domain transfer pairs:

```text
Restaurant -> Laptop
Laptop -> Restaurant
Restaurant -> Twitter
Twitter -> Restaurant
Laptop -> Twitter
Twitter -> Laptop
```

Baselines:

```text
Source-only
Target-supervised upper bound
C3DA source-only
C3DA with target fragments
C3DA + NLI filter
C3DA + NLI filter + label composition
Optional: GCN-enhanced pseudo-labeler
```

Metrics:

```text
Accuracy
Macro-F1
```

For extraction-style experiments, use:

```text
Micro-F1
```

## 9. Key Design Decisions We Made

Important conclusions from our discussion:

1. T5 should remain the generator, not necessarily the final classifier.
2. BERT/RoBERTa remains the final ABSA classifier for fair comparison with C3DA.
3. RSDA's NLI filtering is very suitable for reducing generated-data noise.
4. DAEGCN's domain-specific segment awareness is better than naive high-frequency word extraction.
5. DAEGCN's GCN is worth referencing for pseudo-label quality, but should be implemented later.
6. The first practical version should be:

```text
domain-specific fragment extraction
+ T5/LoRA cross-domain generation
+ NLI filtering
+ C3DA contrastive training
```

## 10. If This Document Is Used Later

If this document is pasted into a future conversation, the intended next step is likely one of:

```text
1. Implement extract_domain_fragments.py
2. Implement nli_filter.py
3. Modify C3DA generation for source_dataset -> target_dataset
4. Design experiments for cross-domain ABSA
5. Write a method section for a paper/proposal
```

The central idea to preserve is:

```text
Build a cross-domain C3DA framework by using DAEGCN-style domain-specific fragment extraction to choose target-domain aspects/opinions, RSDA-style pseudo-labeling and NLI filtering to control generated-data quality, and original C3DA contrastive learning to train a robust final ABSA classifier.
```

