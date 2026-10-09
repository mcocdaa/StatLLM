# 📊 StatLLM: 大语言模型黑盒统计指纹与归因系统

> **基于离散多项分布偏置、狄利克雷平滑似然、中心极限定理置信收敛与 2D PCA 聚类投影的黑盒大模型溯源系统**  
> *覆盖 15 款顶尖 AI 厂商旗舰基座，收录 1,190 条全真实 API 采样指纹，支持复杂 Prompt 扰动抗噪测试。*

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://www.docker.com/)
[![CI](https://github.com/mcocdaa/StatLLM/actions/workflows/ci.yml/badge.svg)](https://github.com/mcocdaa/StatLLM/actions/workflows/ci.yml)
[![Docker Publish](https://github.com/mcocdaa/StatLLM/actions/workflows/docker.yml/badge.svg)](https://github.com/mcocdaa/StatLLM/actions/workflows/docker.yml)

[English](README.md) | [中文文档](README_CN.md)

---

## 🌟 核心理念与技术突破

在当前大模型生态中，各类聚合网关与中转平台层出不穷，部分服务存在**偷梁换柱（Model Swapping）**、**量化暗降（Silent Quantization）** 或 **虚假宣传底座** 的行业乱象。传统的文体学检测受限于长文本内容差异，而白盒水印检测又必须依赖服务商开放隐藏层权重或 Logprobs。

**StatLLM 证明：完全无需内部权重或 Logits，仅凭多道极简离散探针即可精准锁定黑盒基座。**  
自回归 Transformer 模型天然缺乏物理级真随机性，其分词器词表（Tokenizer）、预训练语料频次与对齐微调在底层留下了极其稳固的“数字指纹”：
- **纯粹黑盒非侵入**：仅输入标准 Prompt 并解析输出文本数组，对商业闭源 API 零门槛适配；
- **覆盖 15 款全球顶尖旗舰**：包含 DeepSeek、智谱 GLM、Meta 1.3、Google Gemma 4 / Gemini Pro、OpenAI、Anthropic、xAI、阿里、MiniMax、月之暗面等；
- **工业级抗噪抗扰动**：底库收录 503+ 条包含角色设定（Persona）、多轮闲聊（Chit-chat）、长业务文本噪音（Context Noise）的复合扰动真实样本；
- **严谨概率推断体系**：提供狄利克雷平滑后验概率、均匀零假设基准吻合度（Fitness）以及 68% / 95% 置信区间；
- **样本量收窄智能诊断**：根据误差传播定律 ($SE \propto 1/\sqrt{n}$) 实时评估当前样本量，并在用户提交题目较少时给出量化收窄建议。

---

## 🔬 实测底库模型全景（15 款顶尖模型，1,190 条真实样本）

所有基准样本均直接通过真实 API 物理调用生成，并在不同采样发散度（$T \in [0.70, 0.95]$）下高压检验：

| 厂商 / 机构 | 核心旗舰模型 | 调度通道 / 端点 | 样本量 | 核心离散指纹与统计特征 |
|:---|:---|:---|:---:|:---|
| **深度求索** | **DeepSeek-V4.1-Flash** | `openrouter` | **105** | 整数极强偏好 7、23、41；全排列逆序数分布具有鲜明特征 |
| **智谱 AI** | **GLM-5.3** | `opencode` / `openrouter` | **74** | 偏爱数字 27、84、15；颜色探针中紫/青概率极高，排他性显著 |
| **Meta AI** | **Meta Muse-Spark 1.3** | `opencode` | **69** | 偏好 27、84、33；全排列显著倾向以 4、5 开头 |
| **Meta AI (开源)**| **Llama-3.3-70B** | `openrouter` | **74** | 严格遵循格式指令；石头剪刀布具有鲜明的循环转移特征 |
| **谷歌 (Google 4代)**| **Gemma 4 31B** | `openrouter` | **60** | 4 代最新架构；整数极度扎堆 23、87、12、56、91 |
| **谷歌 (Google 闭源)**| **Gemini Pro (Latest)**| `openrouter` | **70** | 带思维链 CoT；整数高频出现 42、17、88、5、73 |
| **谷歌 (Google 闪电)**| **Gemini 2.5 Flash** | `openrouter` | **85** | 极低延迟响应；英文字母聚类紧贴 G、P、K、D |
| **OpenAI** | **GPT-6-Astra** | `openrouter` | **85** | 整数高频 17、42、64、92；字母偏爱 B、Q、L |
| **OpenAI** | **GPT-5.6-Luna** | `openrouter` | **80** | 整数首词偏好 17（概率超 90%）；确定性模式显著 |
| **OpenAI** | **GPT-6-Luna** | `openrouter` | **60** | 兼顾数学严谨性与确定性偏好 |
| **Anthropic** | **Claude-Sonnet-5.5** | `openrouter` | **85** | 颜色与整数离散熵高，但首位色彩显著偏蓝/青 |
| **xAI** | **Grok-4.7** | `openrouter` | **90** | 偏爱数字 23、37、82；博弈论探针中石头占比显著偏高 |
| **阿里巴巴** | **Qwen-3.8-Max** | `openrouter` | **85** | 中文原生表征强；颜色探针中青/橙占优，聚类紧凑 |
| **MiniMax** | **MiniMax-M3** | `openrouter` | **85** | 词表对齐边界独特；颜色多以橙、紫起手 |
| **月之暗面** | **Kimi-K3** | `openrouter` | **83** | 长上下文抗噪能力极强；字母高频出现 F、L、T、Z、Q |

---

## 🎯 5 大标准离散数学探针

1. **`arr_int5` (随机整数偏好探针)**：要求输出包含 5 个在 1 到 100 之间随机整数的 JSON 数组。捕获分词器数值频次与避免重复倾向。
2. **`arr_color5` (彩虹七色有限集探针)**：在七种标准颜色中随机挑选 5 次。挖掘多项分类词汇的固有次序偏好。
3. **`arr_rps5` (博弈论剪刀石头布探针)**：随机进行 5 局石头剪刀布。揭示模型对非传递博弈选择的对称破缺。
4. **`arr_perm5` (全排列置换探针)**：对 $[1, 2, 3, 4, 5]$ 进行随机打乱。衡量逆序数分布（Inversion Count）与不动点概率。
5. **`arr_letter5` (英文字母离散探针)**：随机输出 5 个大写英文字母。穿透分词器边界，展现字母表离散频率偏向。

---

## 🚀 快速上手 (Quickstart)

### 方式一：Docker Compose 容器部署（推荐）

```bash
# 1. 克隆代码仓库
git clone https://github.com/mcocdaa/StatLLM.git
cd StatLLM

# 2. 一键启动服务
docker compose up -d

# 3. 检查容器运行状态
docker compose ps
docker compose logs -f
```
在浏览器打开 **`http://localhost:8008`** 即可使用。

### 方式二：本地 Python 环境启动

```bash
# 创建并激活虚拟环境
python3 -m venv .venv
source .venv/bin/activate

# 安装依赖及本地包
pip install -r requirements.txt
pip install -e .

# 运行单元测试
pytest -v tests/

# 启动 Web 服务 (指定端口 8008)
statllm serve --port 8008
```

---

## 💻 命令行 CLI 用法

```bash
# 查看题库列表与标准 Prompt
statllm probes

# 查看数据库 15 款模型样本量与统计分布
statllm stats

# 命令行单次直接评测
statllm eval --probe arr_int5 "[42, 17, 88, 5, 73]"
```

---

## 🌐 Web 端核心功能

- **🧪 盲测与鉴定实验室**：支持一键复制 Prompt、批量添加题目卡片、同题多次追加。
- **📊 森林图不确定性分析**：展示 68% 核心置信区间与 95% 全幅置信区间，带详细数值 Hover 提示。
- **🗺️ 2D PCA 聚类散点图**：直观展示 15 款模型在高维多探针空间的相对距离，并以“★ 您当前测试点”精确定位。
- **💡 智能数据量诊断卡片**：实时量化样本量不足对置信区间的扩张影响，并建议精准收窄幅度。
- **🌐 纯净双语无缝切换**：中文与英文模式经过 Playwright 自动化测试严格审查，实现 0 字符污染。

---

## 🔒 隐私与开源安全保障

- **零密钥提交**：代码库经过多层自动化深度扫描，绝对不包含任何明文 API Key。
- **配置示例化**：如需运行扩充评测脚本，请将 [`.env.example`](.env.example) 复制为 `.env` 并填入个人私有 Key（已在 `.gitignore` 彻底忽略）。
- 漏洞报告与披露规范请参阅 [SECURITY.md](SECURITY.md)。

---

## 📄 开源许可证

本项目遵循 [MIT License](LICENSE)。
