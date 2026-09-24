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

按房间编号在厨房、客厅、卧室、卫生间各自划分：前 24 个房间训练，25–27 验证，28–30 内部测试。三个集合场景完全隔离，选择顺序由固定种子 `20260925` 决定。

官方 `test_809.json` 中对应基础任务的记录单独保存，只用于后续在线模拟器评测，不并入训练集。

## 下载边界

远端图像按任务类型保存为 ZIP。下载前审计生成：

- `manifests/train.jsonl`
- `manifests/val.jsonl`
- `manifests/test.jsonl`
- `manifests/official_test.jsonl`
- `remote_files.json`
- `download_plan.json`
- `audit_report.json`

这些文件位于 `/root/pc/gpc_dynaloop_storage/datasets/er_base_subset_v1`，只包含元数据和引用，不包含正式训练图像。

推荐按 ZIP Range 选择性提取清单中的图像；整包顺序下载并删除临时 ZIP 作为兼容回退。下载必须锁定审计报告中的 Hugging Face revision 与 LFS SHA-256。

## 下载前门禁

```bash
bash scripts/run_er_predownload_checks.sh
```

必须满足：9,390 条源训练记录可解析、图像标记与引用一致、配额可满足、场景划分无交集、全部所需归档存在、磁盘预算足够，并且输出目录尚无图像或 ZIP。
