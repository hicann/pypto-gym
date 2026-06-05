# 常见问题与错误恢复

## 常见问题

### Q1: transformers报错 "PyTorch >= 2.4 is required"
升级torch和torch-npu到2.7.1

### Q2: torch_npu报错 "undefined symbol"
确保torch和torch-npu版本完全一致

### Q3: 网络问题或模型加载慢
```bash
export HF_ENDPOINT=https://hf-mirror.com
```
使用 `local_files_only=True` 从本地加载

### Q4: 导入错误（ImportError/FileNotFoundError）
检查导入语句：transformers模块用绝对导入，本地configuration用相对导入

### Q5: model.generate() 触发 CANN ops 缺失崩溃

**症状**：模型加载成功，但 `model.generate()` 时 segfault 或 `EZ9999: Op XXX does not has any binary`（常见缺失：`aclnnEqual`、`aclnnInplaceFillScalar`、`aclnnIsIn`、`aclnnCast`、`aclnnEmbedding`、`aclnnPow`）。

**根因**：CANN ops 内核包不完整或 multi-version 冲突。`npu-smi info` 只检查驱动。

**优先排查**：环境中可能有多个 CANN 安装（如 `cann-8.5.0` / `ptol/cann-8.5.0` / `cann-9.0.0`），`set_env.sh` 指向的版本 ops 不完整。尝试：
```bash
# 检查所有可用 CANN
find /mnt/workspace/gitCode/cann /home/developer -name "set_env.sh" -path "*/ascend*" 2>/dev/null | while read f; do
    echo "=== $f ==="; source "$f" 2>/dev/null
    python3 -c "import torch,torch_npu;torch.npu.set_device(0);x=torch.randn(2,2,dtype=torch.float16).to('npu:0');y=x.to(torch.float32);print('OK')" 2>&1 | head -1
done
```
找出一条输出 `OK` 的路径即是正确的 `set_env.sh`。**三路并行验证确认**：错误路径 → 100% 复现 ops 缺失崩溃。

**临时绕过**：
- `aclnnEqual`（`tie_weights` 触发）：monkey-patch `torch.equal` → CPU compare
- `aclnnInplaceRandom`/`aclnnIsIn`（采样触发）：`do_sample=False`
- `aclnnCast(fp16→bf16)`：CPU 端 cast → NPU

**根治**：确认正确的 CANN `set_env.sh` 路径，或安装匹配的 ops-kernel 包。

### Q6: PyPTO JIT kernel "Not npu device" 错误

**根因**：PTO 注入（`USE_PTO_XXX=True`）发生在模型仍在 CPU 上时，modeling 路由将 CPU tensor 送入 NPU-only JIT kernel。

**修复**：PTO 注入必须在模型加载到 NPU **之后**执行。即 `model.to('npu')` 先于 `_pto.USE_PTO = True`。

### Q7: NPU 上下文污染（单次崩溃导致后续全部 segfault）

**症状**：任意 NPU op 崩溃后，所有后续 NPU 操作（含基线）全部 segfault。

**恢复**：需重置 NPU（重驱/重启），无法通过 Python 恢复。开发时每个 op 异常都要 try-catch 兜底，避免脏状态扩散。

### Q8: FP16 kernel 首次 JIT 编译崩溃但重试通过

**症状**：FP16 kernel 首次运行报 `MPU address access is invalid` 或 `std::bad_alloc`，第二次运行正常。

**根因**：PyPTO JIT 的 FP16 `valid_shape` 尾块处理首次编译时可能存在缓存竞争。8路并行验证中 2/8 路出现。

**修复**：重试即可。若持续出现，减小 `tile_rows`（`8→4`）或用 `pypto.loop_unroll` 替代 `pypto.loop` + `unroll_list`。

## 错误恢复

### 下载中断恢复

若模型下载中断（网络问题、进程被杀），可重新执行下载：
```bash
# 检查已下载文件大小
du -sh {model_weight_dir}/

# 重新下载（resume模式会跳过已下载文件）
export HF_ENDPOINT=https://hf-mirror.com
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='{repo_id}', local_dir='{model_weight_dir}', max_workers=4)
"
```

### 导入修改失败恢复

若导入修改导致语法错误，可从备份恢复：
```bash
# 查看备份文件
ls -la {model_weight_dir}/core/*.bak

# 从备份恢复
cp {model_weight_dir}/core/modeling_{model_type}.py.bak {model_weight_dir}/core/modeling_{model_type}.py
cp {model_weight_dir}/core/configuration_{model_type}.py.bak {model_weight_dir}/core/configuration_{model_type}.py

# 重新使用脚本修复
python3 {pypto_repo}/.agents/skills/pypto-fused-op-integration/scripts/fix_imports.py {model_weight_dir}/core/modeling_{model_type}.py
```
