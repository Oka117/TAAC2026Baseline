# suggestion.pdf 内容归纳

## 一、文档主题

该 PDF 是一份面向推荐/检索类建模流程的优化建议，重点集中在四个方面：

1. 数据清洗：过滤低频用户、低频物品和明显噪声样本。
2. 特征工程：新增时间延迟特征，并对数值特征与多值特征做统一处理。
3. 缺失值处理：删除缺失严重的用户/物品特征，补全可用的整数特征。
4. 模型结构简化：将复杂的原始特征、序列特征和生成模块链路，简化为更直接的 LightGCN 图建模方案。

## 二、具体建议

### 1. 过滤稀疏用户和物品

建议删除交互频次过低的用户和物品：

- 删除 `frequency(user_id) < 5` 的用户。
- 删除 `frequency(item_id) < 5` 的物品。

目的：

- 减少极低频样本带来的噪声。
- 提升用户和物品表示的稳定性。
- 降低长尾极端稀疏数据对模型训练的干扰。

### 2. 构造时间延迟特征并过滤噪声

新增时间延迟特征：

```text
delay_feat = timestamp - label_time
```

同时删除异常或噪声样本：

- 删除 `delay_feat < 3` 的数据。

理解：

- `delay_feat` 用来刻画样本行为时间与标签时间之间的间隔。
- 过小的时间间隔可能代表标签泄漏、行为过近、统计不稳定或业务上不合理的样本，因此建议过滤。

### 3. 处理数值特征和多值特征

对所有 dense numerical feature 进行归一化：

- 包括普通稠密数值特征。
- 也包括 sequence side feature 中的稠密数值特征。

对 multi-value/list 类型特征做平均池化：

- 包括普通多值特征。
- 也包括 sequence side feature 中的多值特征。

理解：

- 归一化可以减少不同量纲、不同取值范围对模型训练的影响。
- 平均池化可以把不定长 list 特征转换成固定维度表示，便于进入后续模型。

### 4. 缺失值处理

建议按缺失比例筛掉质量较差的特征：

- 删除缺失比例超过 `70%` 的 `item_features`。
- 删除缺失比例超过 `70%` 的 `user_features`。

对整数特征中的缺失值进行均值填充：

- 对 `int_feats` 中的 missing value 使用平均值替换。

理解：

- 高缺失比例特征通常信息量不足，保留可能引入噪声。
- 对仍然保留的整数特征，均值填充是一种简单稳定的 baseline 处理方式。

## 三、模型结构简化建议

PDF 第 2 页展示了一个从复杂结构向 LightGCN 简化的思路。

原始链路中涉及的模块/特征包括：

- NS raw feature
- User Embedding
- Item Embedding
- Sequence feature
- Longer NS Token Sequence
- Query
- Generation Module
- Global token
- Hyformer

简化后的核心思路：

- 以每一行样本作为建图或训练输入。
- 使用 `item features` 和 `user features` 表示节点或节点属性。
- 将 `delay_feat` 作为 edge contribution，即边上的贡献/权重/时间相关信号。
- 使用 LightGCN 进行图建模。

理解：

- 原结构包含序列 token、Query、生成模块和 Hyformer，链路较长，工程复杂度较高。
- 建议转向 LightGCN，是为了更直接地建模用户-物品交互关系。
- `delay_feat` 可以为用户-物品边提供时间间隔相关的权重或辅助信息，使图模型不仅看交互是否存在，也能利用交互时间信息。

## 四、可执行落地清单

1. 统计 `user_id` 和 `item_id` 频次，过滤频次小于 5 的用户和物品。
2. 计算 `delay_feat = timestamp - label_time`。
3. 删除 `delay_feat < 3` 的样本。
4. 对所有稠密数值特征做归一化，范围包括 sequence side feature。
5. 对 list/multi-value 特征做 average pooling，范围包括 sequence side feature。
6. 统计 `item_features` 和 `user_features` 的缺失比例，删除缺失比例超过 70% 的特征。
7. 对保留下来的 `int_feats` 缺失值使用均值填充。
8. 梳理用户、物品、交互边三类输入，尝试用 LightGCN 替代复杂的 Hyformer/Generation Module 链路。
9. 将 `delay_feat` 设计为边特征、边权重或边贡献项，并通过验证集比较是否带来收益。

## 五、总体结论

这份 suggestion 的核心不是添加更复杂的模型，而是先把数据质量、特征尺度、缺失值和稀疏交互问题处理干净，再把模型结构从复杂的特征生成/序列建模链路简化为以用户-物品图为中心的 LightGCN 方案。它更像是一套稳健 baseline 的改造建议：先减少噪声和工程复杂度，再用图结构捕捉用户与物品之间的协同关系。

