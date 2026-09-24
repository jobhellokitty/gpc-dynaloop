# GPC-DynaLoop

面向动态开放场景的具身推理项目。当前阶段建立可复现的服务器环境与 AI2-THOR `CloudRendering` 基线。

## 路径

- 代码：`/root/pc/gpc_dynaloop`
- 大文件：`/root/pc/gpc_dynaloop_storage`
- 复用环境：`/root/pc/conda-envs/gqa-process-consistency`
- 参考代码：`/root/pc/Embodied-Omni/embodied_reasoner`

## 基线检查

```bash
bash scripts/capture_environment.sh
bash scripts/run_ai2thor_smoke.sh
```
