# 本地自测说明

## 文件说明

- `self_check.py`：用于检查参赛算法的接口及输出参数是否符合赛题要求。
- `solution/solution.py`：参赛代码提交模板，选手需要在其中实现赛题要求的 6 个接口函数。
- `mini_sample/`：用于本地调试和自测的小样本数据，包含 `linear.pt` 和 `attn.pt`。

---

## 运行环境说明

### 1. 数据样例说明

`mini_sample` 中包含问题 1 和问题 2 各一组完整样例数据。

#### 问题 1：Linear

文件：

```text
mini_sample/linear.pt
```

其中包含 **1 组 Linear 数据**。每组数据为一个字典，主要包含：

```text
weight
calib_activation_list
test_activation_list
```

其中：

- `weight`：Weight（矩阵 B）的 NVFP4 量化数据及对应 scale；
- `calib_activation_list`：用于离线校准的 Activation（矩阵 A）数据，共 **5 组**，每组均为 NVFP4 量化数据及对应 scale；
- `test_activation_list`：用于在线动态量化测试的 Activation 数据，共 **5 组**，每组均为 NVFP4 量化数据及对应 scale。

Linear 自测流程为：

```text
Weight + 5组 Calibration Activation
        ↓
hif4_calibration_and_quantize_weight
        ↓
HiF4 Weight 参数 + activation_state
        ↓
5组 Test Activation
        ↓
hif4_dynamic_quantize_activation
        ↓
检查输出 HiF4 参数是否合法
```

#### 问题 2：Attention

文件：

```text
mini_sample/attn.pt
```

其中包含 **1 组 Attention 数据**。每组数据为一个字典，主要包含：

```text
q_num_heads
kv_num_heads
head_dim
calib
test
```

其中：

- `q_num_heads`：Q 的 attention head 数；
- `kv_num_heads`：K/V 的 attention head 数；
- `head_dim`：每个 attention head 的维度；
- `calib`：用于离线校准的 Q/K/V 数据，共 **5 组**；
- `test`：用于在线动态量化测试的 Q/K/V 数据，共 **5 组**。

每组 Q、K、V 均包含 NVFP4 量化数据及对应 scale。

Attention 自测流程为：

```text
5组 Calibration Q/K/V
        ↓
hif4_calibration_attention
        ↓
q_state / k_state / v_state
        ↓
5组 Test Q/K/V
        ↓
hif4_dynamic_quantize_q
hif4_dynamic_quantize_k
hif4_dynamic_quantize_v
        ↓
检查输出 HiF4 参数是否合法
```

---

### 2. 执行自测脚本

无需修改 `self_check.py` 中的数据路径，直接通过命令行指定 `solution.py` 所在目录和 `mini_sample` 数据目录。

假设目录结构如下：

```text
submission/
├── self_check.py
├── solution/
│   └── solution.py
└── mini_sample/
    ├── linear.pt
    └── attn.pt
```

执行命令：

```bash
python self_check.py \
    --solution_dir ./solution \
    --datasets_dir ./mini_sample
```

Windows 示例：

```powershell
python .\self_check.py --solution_dir .\solution --datasets_dir .\mini_sample
```

其中：

- `--solution_dir`：指定包含 `solution.py` 的目录；
- `--datasets_dir`：指定包含 `linear.pt` 和 `attn.pt` 的 `mini_sample` 目录。

当所有接口、Calibration State 以及 HiF4 输出参数均符合要求时，脚本最终输出：

```text
ALL OUTPUT-FORMAT CHECKS PASSED
```

> `self_check.py` 仅用于本地检查参赛接口及输出格式是否合法，不代表正式评测结果。

已有第三方库：（若选手希望增加第三方库，可提交相关需求，待项目组评审通过后可增添到服务器中）
archspec==0.2.5
backports.zstd==1.5.0
boltons==25.0.0
Brotli==1.2.0
certifi==2026.5.20
cffi==2.0.0
charset-normalizer==3.4.7
conda==26.3.2
conda-libmamba-solver==26.4.2
conda-package-handling==2.4.0
conda_package_streaming==0.12.0
cuda-bindings==13.3.1
cuda-pathfinder==1.5.6
cuda-toolkit==13.0.3.0
distro==1.9.0
filelock==3.29.7
frozendict==2.4.7
fsspec==2026.6.0
h2==4.3.0
hpack==4.1.0
hyperframe==6.1.0
idna==3.17
Jinja2==3.1.6
jsonpatch==1.33
jsonpointer==3.1.1
libmambapy==2.5.0
MarkupSafe==3.0.3
menuinst==2.4.2
mpmath==1.3.0
msgpack==1.1.2
networkx==3.6.1
numpy==2.5.1
nvidia-cublas==13.1.1.3
nvidia-cuda-cupti==13.0.85
nvidia-cuda-nvrtc==13.0.88
nvidia-cuda-runtime==13.0.96
nvidia-cudnn-cu13==9.20.0.48
nvidia-cufft==12.0.0.61
nvidia-cufile==1.15.1.6
nvidia-curand==10.4.0.35
nvidia-cusolver==12.0.4.66
nvidia-cusparse==12.6.3.3
nvidia-cusparselt-cu13==0.8.1
nvidia-nccl-cu13==2.29.7
nvidia-nvjitlink==13.3.33
nvidia-nvshmem-cu13==3.4.5
nvidia-nvtx==13.0.85
packaging==26.2
pip==26.1.2
platformdirs==4.10.0
pluggy==1.6.0
pycosat==0.6.6
pycparser==3.0
PySocks==1.7.1
requests==2.34.2
ruamel.yaml==0.18.17
ruamel.yaml.clib==0.2.15
setuptools==82.0.1
sympy==1.14.0
torch==2.13.0
tqdm==4.67.3
triton==3.7.1
truststore==0.10.4
typing_extensions==4.16.0
urllib3==2.7.0
zstandard==0.25.0
