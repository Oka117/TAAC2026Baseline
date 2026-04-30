# suggestion.pdf 忠实摘要与结构理解（修正版）

## 0. 修正说明

上一版摘要中把第 2 页的 `Light GCN` 理解成“替代 HyFormer”的主干方案，这是不准确的。

根据 PDF 第 2 页 layout 抽取结果，以及你补充的说明，更准确的理解是：

- 后端主干仍然保留 HyFormer。
- `Sequence feature` 仍然先经过 `Longer`，再形成 `Sequence token`。
- `NS raw feature`、`User features`、`Item features` 和 `Delay_feat as edge contributes` 这一侧，是对前端非序列特征和 item/user interaction 做增强。
- `Light GCN` 更像是用于生成或增强 user/item interaction 表征，然后进入 `NS Token` 或相关非序列 token。
- 后续仍然经过 `Query Generation Module`、`Global token` 和 `Hyformer`。

因此，这份 suggestion 的核心不是“把 HyFormer 换成 LightGCN”，而是“在 HyFormer 之前，对非序列特征、序列特征和 user-item/item interaction 进行更强的预处理与表征变换”。

## 1. PDF 原文逐页转写

### Page 1

```text
1, filtering rare item and user

         Delete frequency(user_id)<5    frequency(item_id)<5

2, get new feature

               Delay_feat=timestamp -  label_time

      Filter noise data

               Delete delay_feat<3



3,  all dense numerical feature (including sequence side feature)  do normalization

  Multi value features(list) ( including sequence side feature) do  average pooling



4, handle missing value

         Delete missing proportion (item_features)>70%

         Delete missing proportion (user_features)>70%

         Replace Missing value in int_feats  using average value
```

### Page 2

```text
          5, Simplify the structure


                 NS raw feature                Light GCN                           Sequence feature

                            Every row

   User features                   Item features

    User           Delay_feat as         ItemEmbed                                        Longer
Embedding                                     ding
                   edge
                   contributes


                                                           Query                        Sequence
                    NS Token                        Genration Module                       token





                                                         Global token




                                                           Hyformer
```

说明：PDF 中 `Genration Module` 和 `ItemEmbedding` 的换行来自原始版面，本文保留其含义，按 `Query Generation Module` 和 `Item Embedding` 理解。

## 2. 文档完整内容归纳

PDF 一共提出 5 类建议：

1. 过滤低频 user 和 item。
2. 构造 `Delay_feat` 并过滤噪声数据。
3. 对 dense numerical feature 做归一化，对 multi-value/list 特征做 average pooling。
4. 处理缺失值，包括删除高缺失特征和填补 int 特征缺失值。
5. 简化前端结构，将非序列 raw feature、user/item feature、delay edge contribution 和 LightGCN/item interaction 表征结合起来，最终仍然接入 Longer、Query Generation Module、Global token 和 HyFormer。

## 3. 逐条解释

### 3.1 Filtering rare item and user

原文：

```text
Delete frequency(user_id)<5    frequency(item_id)<5
```

含义：

- 统计每个 `user_id` 的出现频次。
- 统计每个 `item_id` 的出现频次。
- 删除频次小于 5 的用户和物品相关样本或实体。

目标：

- 减少极低频用户/物品带来的稀疏噪声。
- 提高 user embedding 和 item embedding 的训练稳定性。
- 让后续 interaction 或 LightGCN 表征更容易学习到可靠邻居关系。

需要注意：

- 原文没有说明是删除样本、删除节点，还是仅把低频 ID 合并到特殊 bucket。
- 原文也没有说明频次统计范围，是训练集内统计，还是全量统计。

### 3.2 Get new feature: Delay_feat

原文：

```text
Delay_feat=timestamp -  label_time
```

含义：

- 新增一个时间差特征 `Delay_feat`。
- 计算方式是样本 `timestamp` 减去 `label_time`。

在第 2 页中，这个特征又被放到结构图里：

```text
Delay_feat as edge contributes
```

因此更完整的理解是：

- `Delay_feat` 不只是一个普通 dense feature。
- 它还可能被用于 user-item interaction 或 LightGCN 图边上的贡献项、边权重、边属性或边强度。

需要注意：

- PDF 没有明确 `label_time` 的业务含义。
- PDF 没有说明 `Delay_feat` 是直接输入 HyFormer，还是只用于前端 LightGCN/item interaction 特征生成。
- 结合第 2 页版面，更像是把 `Delay_feat` 用在 edge contribution 上，辅助构造 item interaction 信息。

### 3.3 Filter noise data

原文：

```text
Delete delay_feat<3
```

含义：

- 在构造 `Delay_feat` 后，删除 `delay_feat < 3` 的数据。
- 原文将这一步归为 `Filter noise data`。

可能意图：

- 删除时间差过小的异常样本。
- 删除可能存在标签时间异常、行为时间异常或统计不可靠的样本。
- 避免过近时间差对 edge contribution 或模型训练造成噪声。

需要注意：

- 原文没有说明时间单位。
- 原文没有解释为什么阈值是 3。
- 如果 `Delay_feat` 使用的是 `label_time`，必须先确认不会引入标签泄漏。

### 3.4 Dense numerical feature normalization

原文：

```text
all dense numerical feature (including sequence side feature) do normalization
```

含义：

- 所有稠密数值特征都要做归一化。
- 范围包括普通 dense numerical feature。
- 范围也包括 sequence side feature 中的 dense numerical feature。

目标：

- 降低不同数值尺度对模型训练的影响。
- 让 dense feature 更适合作为 NS token、sequence side token 或 interaction feature 的输入。
- 提高优化稳定性。

需要注意：

- 原文没有指定归一化方法，可以是 z-score、min-max、robust scaling、log transform 或分位数截断后归一化。
- 如果 sequence side feature 实际是类别 ID，则不应按 dense numerical feature 归一化。

### 3.5 Multi value features average pooling

原文：

```text
Multi value features(list) ( including sequence side feature) do  average pooling
```

含义：

- 对 list/multi-value 类型特征做平均池化。
- 范围包括普通多值特征。
- 范围也包括 sequence side feature 中的多值特征。

目标：

- 将变长 list 转换成固定维度表示。
- 降低模型输入复杂度。
- 让多值特征可以稳定进入 NS token、sequence token 或前端 interaction 模块。

需要注意：

- 原文没有区分多值类别 ID 和多值 dense 数值。
- 对类别 ID，通常应先 embedding，再对 embedding 做 pooling，而不是直接对 ID 数值求平均。
- 对 aligned int-dense list，简单平均可能损失元素对应关系。

### 3.6 Handle missing value

原文：

```text
Delete missing proportion (item_features)>70%
Delete missing proportion (user_features)>70%
Replace Missing value in int_feats  using average value
```

含义：

- 删除缺失比例超过 70% 的 item features。
- 删除缺失比例超过 70% 的 user features。
- 对 `int_feats` 中的缺失值使用平均值替换。

目标：

- 去掉信息覆盖率过低的特征。
- 避免高缺失特征给模型带来噪声。
- 对保留的 int 特征做缺失补齐。

需要注意：

- 原文没有说明缺失比例是在训练集上统计，还是全量数据上统计。
- 原文没有区分 int 特征是连续数值还是类别 ID。
- 对匿名类别 ID，均值填充通常不合理；这一点需要结合数据字段语义判断。

### 3.7 Simplify the structure

第 2 页结构图包含以下元素：

- `NS raw feature`
- `Light GCN`
- `Sequence feature`
- `Every row`
- `User features`
- `Item features`
- `User Embedding`
- `Item Embedding`
- `Delay_feat as edge contributes`
- `Longer`
- `NS Token`
- `Query Genration Module`
- `Sequence token`
- `Global token`
- `Hyformer`

更准确的结构理解如下：

```text
NS raw feature
  -> Every row
  -> User features / Item features
  -> User Embedding / Item Embedding
  -> Delay_feat as edge contributes
  -> Light GCN 或 item interaction 表征增强
  -> NS Token

Sequence feature
  -> Longer
  -> Sequence token

NS Token + Sequence token
  -> Query Generation Module
  -> Global token
  -> HyFormer
```

也就是说：

- `Light GCN` 位于前端非序列特征和 user-item/item interaction 构造部分。
- `Longer` 位于 sequence feature 到 sequence token 的路径上。
- `Query Generation Module`、`Global token` 和 `Hyformer` 仍然保留。
- PDF 的“simplify”更像是简化或重构输入特征流，而不是删除 HyFormer 后端。

## 4. 与当前实现的对应关系

当前 baseline 已有的部分：

- 非序列特征会转成 `NS Token`。
- 序列特征会转成 `Sequence token`。
- 当前代码支持 `seq_encoder_type=longer`。
- 当前模型后端是 HyFormer block。
- 当前多值离散特征已有 embedding 后 mean pooling。
- 当前序列 timestamp 已经通过 `timestamp - event_timestamp` 转成 time bucket。

PDF 建议新增或强化的部分：

- 低频 user/item 过滤或低频处理。
- `Delay_feat` 及其噪声过滤。
- dense numerical feature 归一化。
- sequence side feature 中 dense/list 特征的归一化或 pooling。
- 高缺失 user/item 特征删除。
- int 特征缺失值填补。
- 基于 user features、item features 和 `Delay_feat` 的 LightGCN/item interaction 表征。

## 5. 一句话总结

这份 suggestion 的重点是：先对数据和前端特征做清洗、归一化、缺失处理和多值聚合，再基于 user/item feature 与 `Delay_feat` 构造 item interaction 或 LightGCN 增强表征，最后仍然把增强后的 `NS Token` 与经过 `Longer` 的 `Sequence token` 送入 Query Generation Module、Global token 和 HyFormer。

