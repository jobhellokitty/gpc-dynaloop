# ER-Base-Subset-v1 下载前规范

## 目标

`ER-Base-Subset-v1` 用于把 `GPC-SFT-HE` 适配为基础具身模型。它是 Embodied-Reasoner 数据的任务子集，不要求完整复现论文，也不包含动态外生变化数据。

纳入能力：

- 单物体搜索；
- 容器内搜索；
- 单物体拾取；
- 容器内拾取；
- 单物体拾取并放置；
- 放入容器；
- 少量导航、开容器、拾取和放置失败轨迹。

排除能力：双物体有序搬运、开关电器、无意义重复轨迹及其他不属于第一阶段基础能力的任务。

## 划分与防泄漏

按房间编号在厨房、客厅、卧室、卫生间各自划分：前 24 个房间训练，25–30 测试。两个集合场景完全隔离。

训练集不再设置数量配额，纳入训练场景中全部 13 类基础任务有效记录。测试集纳入测试场景中实际存在的相同任务记录，用于 checkpoint 选择、调参和阶段性评测。

## 下载边界

远端图像按任务类型保存为 ZIP。下载前审计生成：

- `manifests/train.jsonl`
- `manifests/test.jsonl`
- `remote_files.json`
- `download_plan.json`
- `audit_report.json`

这些文件位于 `/root/pc/gpc_dynaloop_storage/datasets/er_base_subset_v1`，只包含元数据和引用，不包含正式训练图像。

推荐按 ZIP Range 选择性提取清单中的图像；整包顺序下载并删除临时 ZIP 作为兼容回退。下载必须锁定审计报告中的 Hugging Face revision 与 LFS SHA-256。

## 下载前门禁

```bash
bash scripts/run_er_predownload_checks.sh
```

必须满足：9,390 条源训练记录可解析、13 类任务均有有效记录、图像标记与引用一致、Train/Test 场景无交集、全部所需归档存在、磁盘预算足够，并且输出目录尚无图像或 ZIP。

## 训练适配

正式 SFT 使用 Embodied-Reasoner 指定的 LLaMA-Factory `embodied-reasoner` 分支，一条轨迹仅编码一次并监督其全部 assistant 决策。`GPC-SFT-HE` 作为初始权重，冻结视觉编码器和多模态投影层，只训练语言模型 attention LoRA。训练时通过 `new_special_tokens` 注册作者分支所需的 `<|feedback|>`，不修改原始 `GPC-SFT-HE` 目录。

```bash
python scripts/export_er_llamafactory_dataset.py
```

训练配置为 `configs/er_base_lora.yaml`。Step 级索引仅用于数据诊断、单步门禁和后续 Agent Loop，不用于正式全量 SFT。
