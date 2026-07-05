# FAL Long-Term Fair Query Experiments

本目录是“面向部分参与联邦主动学习的长期公平查询调度与跨客户端去冗余”的最小实验环境。

当前目标是先服务两周 No-Go：

1. 验证 partial participation 下是否存在长期 query concentration。
2. 验证跨客户端 redundancy 是否稳定出现。
3. 检查 quota / k-center-like memory 等简单 baseline 是否已经足够。

注意：这不是 LoGo、KAFAL、IFAL 或 FairFAL 的官方复现代码。正式论文实验需要接入至少一个公开论文代码库作为强 baseline；本仓库用于先跑通自定义 No-Go 协议和候选方法原型。

## 目录

```text
configs/                 实验配置
scripts/download_datasets.py
scripts/run_no_go.py     最小 No-Go 训练与查询循环
src/fal_experiment/      实验框架源码
```

## 环境

本项目使用 Conda 隔离环境 `te-fal`，并安装 CUDA 12.8 版 PyTorch。不要把依赖安装到全局 Python。

```powershell
D:\conda\Scripts\conda.exe create -y -n te-fal python=3.11 pip
D:\conda\Scripts\conda.exe run -n te-fal python -m pip install -r .\requirements.txt
```

GPU 验证：

```powershell
D:\conda\Scripts\conda.exe run -n te-fal python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

## 数据集

数据默认下载到项目根目录的 `data/torchvision/`，不会纳入版本控制。

```powershell
D:\conda\Scripts\conda.exe run -n te-fal python .\scripts\download_datasets.py --datasets FashionMNIST MNIST CIFAR10
```

## Smoke Test

```powershell
D:\conda\Scripts\conda.exe run -n te-fal python .\scripts\run_no_go.py --config .\configs\smoke_fashionmnist.yaml --strategy qfair
```

输出默认写到 `runs/fal_longterm_fair_query/`。

## V2 门禁运行顺序

当前实验按硕士小论文门禁执行，不直接跑完整大表。

最新本地结论见：

- `D:\DeskTop\te\04_validation\no_go_results\fal_fashionmnist_fix_v2_20260705.md`
- `D:\DeskTop\te\05_outputs\paper_plans\fal_longterm_fair_query\03_experiment_design.md`

当前状态仍是 **Narrow / Fix before Go**。不要直接进入 CIFAR-10 主实验；先稳定 Fashion-MNIST 训练，并验证 `quota_entropy`、`quota_red_entropy`、`class_aware_quota_red` 的三 seed trade-off 是否真实。当前 `qfair` 只保留为诊断 baseline。

### G0：协议 sanity

先确认训练协议能学起来：

```powershell
D:\conda\Scripts\conda.exe run -n te-fal python .\scripts\run_no_go.py --config .\configs\sanity_fashionmnist.yaml --strategy entropy
```

### G1：现象诊断

比较 uniform 和 long-tail availability：

```powershell
D:\conda\Scripts\conda.exe run -n te-fal python .\scripts\run_no_go.py --config .\configs\diagnostic_fashionmnist_uniform.yaml --strategy entropy
D:\conda\Scripts\conda.exe run -n te-fal python .\scripts\run_no_go.py --config .\configs\diagnostic_fashionmnist_longtail.yaml --strategy entropy
```

### G2：机制和强反证 baseline

先跑 Fashion-MNIST 全 baseline 矩阵：

```powershell
.\scripts\run_strategy_matrix.ps1 -Config .\configs\diagnostic_fashionmnist_longtail.yaml
```

然后汇总：

```powershell
D:\conda\Scripts\conda.exe run -n te-fal python .\scripts\summarize_runs.py --dataset FashionMNIST --latest-per-strategy --output .\runs\fal_longterm_fair_query\fashionmnist_latest_summary.csv
```

### Fix V2：当前推荐复跑命令

修正后的 Fashion-MNIST Narrow/Fix 配置：

```powershell
.\scripts\run_strategy_matrix.ps1 -Config .\configs\fix_fashionmnist_longtail.yaml -Seeds 7,42,123 -Strategies entropy,quota_entropy,quota_red_entropy,qfair
```

汇总：

```powershell
D:\conda\Scripts\conda.exe run -n te-fal python .\scripts\summarize_runs.py --dataset FashionMNIST --latest-per-strategy --aggregate-seeds --output .\runs\fal_longterm_fair_query\fashionmnist_fix_v2_multiseed_aggregate.csv
```

### Fix V3：下一步实验命令

训练稳定性扫描已确认 Adam lr=0.003 是当前 Fashion-MNIST V3 的默认稳定协议；SGD lr=0.01/0.02/0.05 未通过 E0 gate。确认命令：

```powershell
D:\conda\Scripts\conda.exe run -n te-fal python .\scripts\run_no_go.py --config .\configs\fix_fashionmnist_v3.yaml --strategy entropy --seed 7 --run-tag e0_adam_lr0p003_confirm
D:\conda\Scripts\conda.exe run -n te-fal python .\scripts\run_no_go.py --config .\configs\fix_fashionmnist_v3.yaml --strategy entropy --seed 42 --run-tag e0_adam_lr0p003_confirm
D:\conda\Scripts\conda.exe run -n te-fal python .\scripts\run_no_go.py --config .\configs\fix_fashionmnist_v3.yaml --strategy entropy --seed 123 --run-tag e0_adam_lr0p003_confirm
```

E1/E2/E3 第一轮诊断已完成。若需要复跑本轮诊断：

```powershell
.\scripts\run_strategy_matrix.ps1 -Config .\configs\fix_fashionmnist_v3.yaml -Seeds 7,42,123 -Strategies entropy,quota_entropy,quota_red_entropy,class_aware_quota_red,qfair,red_entropy
```

汇总：

```powershell
D:\conda\Scripts\conda.exe run -n te-fal python .\scripts\summarize_runs.py --dataset FashionMNIST --latest-per-strategy --aggregate-seeds --output .\runs\fal_longterm_fair_query\fashionmnist_fix_v3_multiseed_aggregate.csv
```

当前不要直接迁移到 CIFAR-10。G3 复核后，`quota_red_entropy` 不再作为主 claim；participation stress test 已确认 p=0.5 支持 fairness claim、p=0.1 是失效边界。若需要复跑 stress test，可使用 `--participation-rate`：

```powershell
D:\conda\Scripts\conda.exe run -n te-fal python .\scripts\run_no_go.py --config .\configs\fix_fashionmnist_v3.yaml --strategy entropy --seed 7 --participation-rate 0.5 --run-tag v3_participation0p5
D:\conda\Scripts\conda.exe run -n te-fal python .\scripts\run_no_go.py --config .\configs\fix_fashionmnist_v3.yaml --strategy quota_entropy --seed 7 --participation-rate 0.5 --run-tag v3_participation0p5
```

## 当前限制

- 这是 No-Go 框架，不是 LoGo / KAFAL / IFAL / FairFAL 的完整复现。
- 隐私只做低维摘要设置，不提供 formal privacy proof。
- 模型是轻量 CNN，用于快速判断选题可行性。GTX 1650 只有 4GB 显存，正式实验优先控制 batch size 和候选池大小。
- 论文主实验通过 G2 后，再接入 LoGo 或 FairFAL 官方代码；否则只能写自建协议的 preliminary study。
