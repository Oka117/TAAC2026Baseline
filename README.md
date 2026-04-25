# TAAC2026Baseline

## Source Code Analysis

- [中文源码分析](README.zh.md)
- [English Source Analysis](README.en.md)

## Introduction

**Towards Unifying Sequence Modeling and Feature Interaction for Large-scale Recommendation**

Recommender systems power both large-scale content platforms, such as feeds and short videos, and digital advertising systems, such as CTR/CVR prediction. They directly shape user experience, engagement, and revenue. Operating under massive traffic and strict latency constraints, these systems make billions of real-time decisions daily and underpin a hundred-billion dollar digital advertising market.

Over the past two decades, recommendation research has progressed along two major branches:

- **Feature interaction models**, which focus on modeling high-dimensional multi-field categorical and contextual features.
- **Sequential models**, which capture the temporal dynamics of user behavior through embedding-based retrieval systems and Transformer-style ranking models.

Although both paradigms have achieved significant success, they have largely evolved independently. This separation has created structural bottlenecks in industrial systems: shallow cross-paradigm interaction, inconsistent optimization objectives, limited scalability, and increasing hardware and engineering complexity. As sequence lengths and model scales continue to grow, these fragmented architectures become increasingly inefficient.

In recent years, several works have begun to bridge these two historically separated branches. To further accelerate progress in this direction, the challenge **"Towards Unifying Sequence Modeling and Feature Interaction for Large-scale Recommendation"** encourages participants to develop:

- a **unified tokenization scheme**;
- a **homogeneous, stackable backbone**;
- a single architecture that jointly models sequential user behaviors and non-sequential multi-field features for CVR prediction.

Submissions are ranked by a single **AUC of ROC** metric. Beyond the leaderboard, the competition offers two innovation awards:

- **Unified Block Innovation Award ($45,000)**
- **Scaling Law Innovation Award ($45,000)**

These awards are independent of leaderboard rank. The workshop paper review emphasizes novelty and insight in unified architectures and systematic scaling-law exploration rather than focusing solely on AUC.

## Dataset

The dataset released in this competition is fully anonymized and does not reflect the exact production characteristics of Tencent's advertising platform.

The dataset is a large-scale industrial dataset constructed from real-world advertising logs. It consists of two main components:

1. **User behavior sequences**
2. **Non-sequential multi-field features**

User behavior sequences contain interaction events between users and items, such as exposure, click, and conversion. Each event is associated with side information such as timestamps and action types.

Multi-field features include:

- user attributes;
- item attributes;
- contextual signals;
- cross features.

To ensure fairness and protect privacy, all sparse features are represented as anonymized integer IDs, and dense features are provided as fixed-length float vectors. No raw content, such as text, image, URL, or personally identifiable information, is released.

The preliminary round dataset contains **200 million user sequences** and uses a **flat column layout**, where all features are stored as individual top-level columns instead of nested structs or arrays.

## Columns

The 120 columns fall into 6 categories.

| Category | Count | Dataset | Description |
| --- | ---: | --- | --- |
| ID & Label | 5 | `int64` / `int32` | Core identifiers, label, and timestamp. |
| User Int Features | 46 | `int64` / `list<int64>` | Discrete user features, including both single-value scalar features, such as age and gender, and multi-value array features, such as marital status, describing user basic attributes and preferences. |
| User Dense Features | 10 | `list<float>` | Continuous-valued user features, including embeddings and other aligned signals for some corresponding integer features. |
| Item Int Features | 14 | `int64` / `list<int64>` | Discrete item features, including item categories, types, other basic information, and multi-label information for items. |
| Domain Sequence Features | 45 | `list<int64>` | Behavioral sequence features from 4 domains. |

## Detailed Column Schema

### ID & Label Columns (5 columns)

All these 5 columns have no null value.

| Column | `user_id` | `item_id` | `label_type` | `label_time` | `timestamp` |
| --- | --- | --- | --- | --- | --- |
| Data Type | `int64` | `int64` | `int32` | `int64` | `int64` |

### User Int Features (46 columns)

- `user_int_feats_{1, 3, 4, 48-59, 82, 86, 92-109}`: scalar `int64`, total 35 columns.
- `user_int_feats_{15, 60, 62-66, 80, 89-91}`: array `list<int64>`, total 11 columns.

### User Dense Features (10 columns)

- `user_dense_feats_{61, 87}`: array `list<float>`, total 2 columns, representing user embedding features such as SUM and LMF4Ads.
- `user_dense_feats_{62-66, 89-91}`: array `list<float>`, total 8 columns, corresponding to the integer features `user_int_feats_{62-66, 89-91}` with the same length.

Example:

```text
user_int_feats_62:   [1, 2, 3]
user_dense_feats_62: [10.5, 20, 15.5]
```

Explanation: the two arrays are aligned element by element. For example, the first element in `user_int_feats_62`, value `1`, denotes a specific entity or category, while the first element in `user_dense_feats_62`, value `10.5`, provides statistics for that element, such as dwell time, a score, or a probability.

### Item Int Features (14 columns)

- `item_int_feats_{5-10, 12-13, 16, 81, 83-85}`: scalar `int64`, total 13 columns.
- `item_int_feats_{11}`: array `list<int64>`, total 1 column.

### Domain Sequence Features (45 columns)

`list<int64>` sequences from 4 behavioral domains:

- `domain_a_seq_{38-46}`: 9 columns.
- `domain_b_seq_{67-79, 88}`: 14 columns.
- `domain_c_seq_{27-37, 47}`: 12 columns.
- `domain_d_seq_{17-26}`: 10 columns.

## Task

The main objective of this competition is to design a unified, stackable modeling block that simultaneously handles multi-field non-sequential tokens and sequential behavior tokens in one architecture.

Each training/test instance corresponds to a triplet:

```text
(user, context, target ad/item)
```

The inputs consist of:

1. **Non-sequential multi-field features**: user, ad, context, and cross features.
2. **User behavior sequence**: chronological user interaction histories with heterogeneous side information.

Participants should build an effective model to capture the correlations between all these features and output a predicted conversion rate, or **pCVR**, for the target ad.

The competition encourages, but does not require, participants to explore:

1. building stackable unified blocks over both sequence and non-sequence features, or tokens;
2. the scaling laws of the model.

The final leaderboard score is the **AUC of ROC** score.

## Competition Framework

The official task page describes the framework as follows.

### Bottom-left: Input

Each instance consists of:

- sequential behavior data, namely chronological user interaction histories with items and timestamps across multiple users;
- non-sequential multi-field features grouped into four categories:
  - User Features;
  - Ads Features;
  - Context Features (sparse);
  - Cross Features (dense embeddings).

All data is subject to privacy protection.

### Right: A Demo Model

The sequential and non-sequential features are converted into:

- `S` tokens by a sequential tokenizer;
- `NS` tokens by a non-sequential tokenizer.

These tokens are then jointly processed by a stack of unified blocks within a single homogeneous backbone, followed by a CVR prediction head.

### Top-left: Training & Evaluation

The model is trained through loop-based optimization with a cross entropy loss:

```text
Loss(y_hat, y) <- Predicted CVR(y_hat)
y in {0 = No Conv., 1 = Conv.}
```

The objective is to maximize:

```text
AUC-ROC
```

The final leaderboard score is the AUC of ROC score.
