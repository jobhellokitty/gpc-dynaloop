# GPC-DynaLoop 轨迹协议 v1.0.0

## 范围

本协议统一表示场景、动作、观测、动作结果和完整轨迹，供 AI2-THOR 数据生成、基础具身训练和后续 Agent Loop 共用。

实现属于对 Embodied-Reasoner 的方法适配，不是其数据格式的严格复现。动作语义参考：

- `/root/pc/Embodied-Omni/embodied_reasoner/data_engine/baseAction.py`
- `/root/pc/Embodied-Omni/embodied_reasoner/data/train_multiturn_9390.json`
- `/root/pc/Embodied-Omni/embodied_reasoner/data/test_809.json`

## 核心对象

| 对象 | 必要内容 |
|---|---|
| `SceneSpec` | 场景 ID、随机种子、模拟器及版本 |
| `ActionCommand` | 唯一动作 ID、动作类型、目标物体及参数 |
| `Observation` | 智能体位姿、可见物体、持有物、图像尺寸、状态摘要 |
| `ActionResult` | 成功标志、错误信息、耗时、动作后观测 |
| `TrajectoryStep` | 连续步骤编号、动作与结果 |
| `Trajectory` | 协议版本、任务、场景、步骤及扩展元数据 |

动作类型为 `observe`、`navigate`、`pickup`、`put`、`move`、`rotate`、`look` 和 `end`。`navigate`、`pickup`、`put` 必须提供 `target_object_id`。

## 状态一致性

`state_digest` 是规范化状态的 SHA-256，覆盖智能体位姿以及全部物体的位置、旋转和关键状态。浮点数保留六位小数；确定性测试比较结构化状态，不要求渲染图像逐字节一致。

## 运行与输出

```bash
bash scripts/run_stage1_protocol_suite.sh
```

结果保存在：

- `/root/pc/gpc_dynaloop_storage/results/stage1/stage1_protocol_suite.json`
- `/root/pc/gpc_dynaloop_storage/results/stage1/stage1_trajectories.jsonl`

通过条件：20 个协议化动作全部成功，4 个固定场景在相同随机种子下状态摘要全部一致。

固定种子比较使用语义状态和显式容差：位置差不超过 `0.01m`、旋转差不超过 `0.1°`，且物体集合、开关、容器和持有状态完全一致。该标准避免把物理引擎的微小浮点沉降误判为环境不确定性。

阶段一完整门禁还会加载合并后的 `GPC-SFT-HE`，处理两张连续 AI2-THOR 图像并生成非空输出：

```bash
bash scripts/run_stage1_full_validation.sh
```

完整门禁结果保存至 `/root/pc/gpc_dynaloop_storage/results/stage1/stage1_gate.json`。
