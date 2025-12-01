# AgentColab - 自动论文处理与创新想法生成系统

> 基于多个大语言模型的自动化论文处理系统，从PDF提取、分析总结、生成创新想法，到自动生成代码实现。

## 📋 目录

- [功能特点](#功能特点)
- [快速开始](#快速开始)
- [使用方式](#使用方式)
- [配置说明](#配置说明)
- [MinerU PDF解析](#mineru-pdf解析)
- [项目结构](#项目结构)
- [常见问题](#常见问题)

---

## 🌟 功能特点

### 核心功能流程

```
PDF文档 → 提取文本 → 清洗内容 → 深度分析 → 生成想法 → 详细化 → 生成代码
```

### 详细功能

1. **PDF文档提取**
   - ✅ MinerU API高精度提取（支持公式、表格、图片）
   - ✅ PyPDF2备选方案
   - ✅ 批量处理支持

2. **智能论文清洗**
   - ✅ 使用Python规则自动清理引用、参考文献、附录等
   - ✅ 保留核心研究内容

3. **深度论文分析（DeepSeek）**
   - ✅ 总结论文核心内容（研究问题、创新点）
   - ✅ 分析核心算法实现逻辑（算法原理、关键步骤）
   - ✅ 提取技术亮点和贡献
   - ✅ 输出Markdown格式的详细分析

4. **创新想法生成**
   - ✅ 基于多篇论文生成创新想法
   - ✅ 自动评分和筛选

5. **想法详细化**
   - ✅ 将想法展开为完整研究方案

6. **代码自动生成**
   - ✅ 使用Claude API生成Python实现代码

---

## 🚀 快速开始

### 1. 环境初始化

```bash
# 进入项目目录
cd Agent_Colab

# 运行初始化
./run.sh setup
```

### 2. 配置API密钥

**方式A: 环境变量（推荐）**

```bash
export GOOGLE_API_KEY="your_gemini_api_key"
export DEEPSEEK_API_KEY="your_deepseek_api_key"
export ANTHROPIC_API_KEY="your_claude_api_key"
export MINERU_API_KEY="your_mineru_api_key"  # 可选
```

**方式B: 配置文件**

编辑 `config.yaml`：

```yaml
api_keys:
  google_api_key: "your_gemini_api_key"
  deepseek_api_key: "your_deepseek_api_key"
  anthropic_api_key: "your_claude_api_key"
  mineru_api_key: "your_mineru_api_key"  # 可选
```

**获取API密钥：**
- Gemini: https://makersuite.google.com/app/apikey
- DeepSeek: https://platform.deepseek.com/
- Claude: https://console.anthropic.com/
- MinerU: https://mineru.net/ （每天2000页免费）

### 3. 准备PDF文件

将PDF论文放入 `data/input/` 目录，或准备PDF的公开URL。

### 4. 开始使用

```bash
# 方式1: Web UI（最简单）
./run.sh ui

# 方式2: 命令行完整流程
./run.sh full

# 方式3: 分步执行
./run.sh pdf      # 提取PDF
./run.sh clean    # 清洗论文
./run.sh analyze  # 分析论文
./run.sh idea     # 生成想法
./run.sh code     # 生成代码
```

---

## 🎨 使用方式

### 方式1: Web UI（推荐 ⭐）

**优点**：图形界面，操作简单，实时反馈

```bash
# 启动Web界面
./run.sh ui

# 或使用
./start_ui.sh
```

浏览器自动打开 http://localhost:7860

**Web UI功能**：
- ⚙️ 配置：API密钥管理
- 📄 PDF提取：单个/批量提取
- 📖 论文处理：清洗和分析
- 💡 想法生成：创新想法生成
- 💻 代码生成：Python代码实现
- 🚀 完整流程：一键执行全部步骤

### 方式2: 命令行

**优点**：快速执行，适合自动化

```bash
# 检查环境
./run.sh check

# 运行完整流程
./run.sh full

# 单步执行
./run.sh pdf        # PDF提取
./run.sh clean      # 论文清洗
./run.sh analyze    # 论文分析
./run.sh idea       # 想法生成
./run.sh select     # 想法筛选
./run.sh detail     # 想法详细化
./run.sh code       # 代码生成
```

### 方式3: Python API

**优点**：完全可定制，集成到其他项目

#### 示例1：使用MinerU提取PDF

```python
from agents import PDFExtractorAgent

# 初始化Agent（使用MinerU）
agent = PDFExtractorAgent(use_mineru=True)

# 方式A：从URL提取
content = agent.extract_from_url(
    pdf_url="https://arxiv.org/pdf/2301.00001.pdf",
    pdf_name="example_paper"
)

# 结果：
# - data/extracted/example_paper_extracted.txt（纯文本）
# - data/extracted/example_paper_mineru/（完整结果）
#   - extracted/full.md（Markdown含公式表格）
#   - extracted/layout.json（布局信息）
#   - extracted/{uuid}_content_list.json（内容列表）
#   - extracted/{uuid}_model.json（模型信息）
#   - extracted/images/（所有图片）

print(f"提取的文本长度: {len(content)} 字符")

# 方式B：上传本地文件
content = agent.extract_from_file(
    pdf_path="path/to/local/paper.pdf",
    pdf_name="local_paper"
)

# 方式C：批量处理URL
urls = [
    "https://example.com/paper1.pdf",
    "https://example.com/paper2.pdf"
]
results = agent.extract_from_urls(
    pdf_urls=urls,
    pdf_names=["paper1", "paper2"]
)
# 返回: {"paper1": "文本内容...", "paper2": "文本内容..."}

# 方式D：批量上传本地文件
files = ["paper1.pdf", "paper2.pdf", "paper3.pdf"]
results = agent.batch_extract_from_files(files)
```

#### 示例2：使用PyPDF2提取本地PDF

```python
from agents import PDFExtractorAgent

# 初始化Agent（使用PyPDF2）
agent = PDFExtractorAgent(use_mineru=False)

# 自动处理data/input/目录下的所有PDF
results = agent.run()

# 结果：
# - data/extracted/{论文名}_extracted.txt（每个PDF一个文件）

for name, content in results.items():
    print(f"{name}: {len(content)} 字符")
```

#### 示例3：完整流程

```python
from main import AgentColab

# 创建实例
agentcolab = AgentColab()

# 运行完整流程
results = agentcolab.run_full_pipeline()
# 依次执行：PDF提取 → 清洗 → 分析 → 想法生成 → 代码生成

# 或单步执行
agentcolab.run_module('pdf_extract')    # 只提取PDF
agentcolab.run_module('paper_analyze')  # 只分析论文
```

#### 示例4：自定义工作流

```python
from agents import PDFExtractorAgent, PaperAnalyzerAgent

# 1. 提取PDF
pdf_agent = PDFExtractorAgent(use_mineru=True)
papers = pdf_agent.extract_from_urls(
    pdf_urls=["https://example.com/paper1.pdf"],
    pdf_names=["paper1"]
)

# 2. 自定义处理
text = papers["paper1"]
# 你的自定义逻辑...

# 3. 分析论文
analyzer = PaperAnalyzerAgent()
results = analyzer.run({"paper1": text})

# 4. 访问结果
analysis = results["paper1"]
print(analysis["summary"])
```

---

## ⚙️ 配置说明

### 配置文件：`config.yaml`

配置文件包含三大部分：API密钥、API参数、流程参数。

#### 1. API密钥配置

**优先级**：环境变量 > `config.yaml` > 空字符串

```yaml
api_keys:
  google_api_key: ""          # Gemini API密钥
  deepseek_api_key: ""        # DeepSeek API密钥  
  anthropic_api_key: ""       # Claude API密钥
  mineru_api_key: ""          # MinerU API密钥（可选）
```

**推荐方式**：使用环境变量
```bash
export GOOGLE_API_KEY="your_key"
export DEEPSEEK_API_KEY="your_key"
export ANTHROPIC_API_KEY="your_key"
export MINERU_API_KEY="your_key"
```

#### 2. API参数配置

**Gemini配置**：
```yaml
api:
  gemini:
    model: "gemini-2.5-flash"        # 模型名称
    temperature: 0.7                  # 随机性（0-1）
    max_output_tokens: 8192           # 最大输出长度
```

**DeepSeek配置**：
```yaml
api:
  deepseek:
    base_url: "https://api.deepseek.com"
    model: "deepseek-chat"
    temperature: 0.7
```

**Claude配置**：
```yaml
api:
  claude:
    model: "claude-3-5-sonnet-20241022"
    temperature: 0.7
    max_tokens: 4096
```

**MinerU配置**：
```yaml
api:
  mineru:
    base_url: "https://mineru.net/api/v4"
    timeout: 300                      # 请求超时（秒）
    model_version: "vlm"              # vlm或pipeline
    enable_formula: true              # 是否提取公式
    enable_table: true                # 是否提取表格
    language: "auto"                  # 语言识别
```

#### 3. 流程参数配置

**PDF提取配置**：
```yaml
pipeline:
  pdf_extraction:
    use_mineru: false                 # 是否使用MinerU
    fallback_to_pypdf2: true          # MinerU失败时回退到PyPDF2
    mineru_model: "vlm"               # MinerU模型：vlm或pipeline
    max_wait_time: 600                # 最大等待时间（秒）
    poll_interval: 5                  # 状态检查间隔（秒）
```

**论文清洗配置**（待实现）：
```yaml
pipeline:
  paper_cleaning:
    enabled: true                     # 是否启用清洗
    remove_references: true           # 移除参考文献
    remove_acknowledgments: true      # 移除致谢
    keep_formulas: true               # 保留公式
    keep_tables: true                 # 保留表格
```

**论文分析配置**（待实现）：
```yaml
pipeline:
  paper_analysis:
    do_translation: true              # 是否翻译为中文
    do_formula_analysis: true         # 是否分析公式
    do_summary: true                  # 是否生成总结
    extract_methods: true             # 是否提取方法
    extract_results: true             # 是否提取结果
```

**想法生成配置**（待实现）：
```yaml
pipeline:
  idea_generation:
    min_ideas: 3                      # 最少生成想法数
    max_ideas: 10                     # 最多生成想法数
    score_threshold: 60               # 最低分数阈值
    creativity_level: 0.8             # 创造性水平（0-1）
```

#### 4. 目录配置

```yaml
directories:
  data_root: "data"                   # 数据根目录
  input: "data/input"                 # PDF输入目录
  extracted: "data/extracted"         # 提取结果目录
  cleaned: "data/cleaned"             # 清洗结果目录
  analyzed: "data/analyzed"           # 分析结果目录
  ideas: "data/ideas"                 # 想法目录
  code: "data/code"                   # 代码目录
  logs: "logs"                        # 日志目录
```

#### 5. 日志配置

```yaml
logging:
  level: "INFO"                       # 日志级别
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  date_format: "%Y-%m-%d %H:%M:%S"
  file_prefix: "agentcolab"           # 日志文件前缀
```

**日志级别说明**：
- `DEBUG`：详细调试信息（开发时使用）
- `INFO`：一般信息（推荐）
- `WARNING`：警告信息
- `ERROR`：错误信息
- `CRITICAL`：严重错误

**日志文件位置**：`logs/agentcolab_YYYYMMDD.log`

---

## 📦 模块详细说明

### 1. PDF提取模块 (`PDFExtractorAgent`)

**状态**：✅ 已完全实现

#### 功能概述
支持两种PDF提取方式：MinerU高精度提取和PyPDF2本地提取。

#### MinerU方式

**提取内容**：
- 📝 文本内容（包括正文、标题、段落）
- 🧮 数学公式（LaTeX格式）
- 📊 表格（Markdown格式）
- 🖼️ 图片（自动提取并保存）
- 📐 布局信息（保持原文档结构）

**支持的输入方式**：
1. **URL方式**：提供PDF的公开URL
2. **文件上传**：通过Web UI直接上传本地PDF文件

**工作流程**：
1. 上传PDF到MinerU服务器（文件上传模式）或提供URL
2. 创建解析任务（自动选择VLM或Pipeline模型）
3. 轮询任务状态（每5秒检查一次）
4. 下载解析结果（ZIP格式）
5. 自动解压并保存

**生成的文件**：
```
data/extracted/
├── 论文名_extracted.txt          # 纯文本内容
└── 论文名_mineru/                 # MinerU完整结果
    ├── extracted/
    │   ├── full.md                # Markdown格式（含公式、表格）
    │   ├── layout.json            # 页面布局信息
    │   ├── {uuid}_content_list.json   # 内容列表
    │   ├── {uuid}_model.json          # 模型识别信息
    │   ├── {uuid}_origin.pdf          # 原始PDF（保留）
    │   └── images/                # 提取的所有图片
    │       ├── {hash1}.jpg        # 图片（hash命名）
    │       ├── {hash2}.jpg
    │       └── ...
    └── result.zip                 # 原始ZIP文件（保留）
```

**配置参数**：
```yaml
pipeline:
  pdf_extraction:
    use_mineru: true              # 是否使用MinerU
    fallback_to_pypdf2: true      # 失败时回退到PyPDF2
    mineru_model: "vlm"           # vlm或pipeline
    max_wait_time: 600            # 最大等待时间（秒）
    poll_interval: 5              # 状态检查间隔（秒）
```

**使用示例**：
```python
from agents import PDFExtractorAgent

# Web UI上传方式
agent = PDFExtractorAgent(use_mineru=True)
content = agent.extract_from_file("path/to/paper.pdf", "论文名称")

# URL方式
content = agent.extract_from_url(
    pdf_url="https://example.com/paper.pdf",
    pdf_name="论文名称"
)

# 批量处理
files = ["paper1.pdf", "paper2.pdf"]
results = agent.batch_extract_from_files(files)
```

#### PyPDF2方式

**提取内容**：
- 📝 纯文本内容（基础文本提取）
- ⚠️ 不支持公式识别
- ⚠️ 不支持表格结构
- ⚠️ 不支持图片提取

**工作流程**：
1. 读取本地PDF文件（`data/input/`）
2. 逐页提取文本
3. 保存为纯文本文件

**生成的文件**：
```
data/extracted/
└── 论文名_extracted.txt          # 纯文本，无格式
```

**使用场景**：
- ✅ 简单文本文档
- ✅ 不需要公式和表格
- ✅ 完全离线处理
- ✅ 无API限额

---

### 2. 论文清洗模块 (`PaperCleanerAgent`)

**状态**：✅ 已完全实现

#### 功能概述
使用Python规则（正则表达式）自动清理论文中的无关内容，保留核心研究内容。

#### 清洗内容

**删除的部分**：
- ❌ References（参考文献）- 支持多种格式标题
- ❌ Acknowledgments（致谢）
- ❌ Appendix（附录）
- ❌ Funding Information（资助信息）
- ❌ Author Contributions（作者贡献）
- ❌ Conflict of Interest（利益冲突声明）
- ❌ 行内引用标记：`[1]`, `[2-5]`, `(Smith et al., 2020)`
- ❌ URL和邮箱地址
- ❌ 页码标记
- ❌ 多余空白行

**保留的部分**：
- ✅ 标题和摘要
- ✅ 核心研究内容
- ✅ 算法和方法
- ✅ 公式和表格
- ✅ 实验结果

#### 工作流程

1. 从 `data/collections/all_papers.json` 读取论文
2. 对每篇论文应用清洗规则
3. 保存清洗后的文本到 `data/cleaned/paper_*_cleaned.txt`
4. 创建清洗后的集合 `data/collections/all_papers_cleaned.json`

#### 生成的文件

```
data/cleaned/
├── paper_1_cleaned.txt          # 清洗后的论文1
├── paper_2_cleaned.txt          # 清洗后的论文2
└── ...

data/collections/
└── all_papers_cleaned.json      # 清洗后的论文集合
```

#### 清洗效果

典型删除率：**5-20%**
- 保留核心内容
- 删除引用和无关章节
- 提高后续分析效率

#### 使用示例

```python
from agents import PaperCleanerAgent

agent = PaperCleanerAgent()
results = agent.run()  # 自动从集合读取并清洗

# 查看统计
for paper_key, content in results.items():
    print(f"{paper_key}: {len(content)} 字符")
```

---

### 3. 论文分析模块 (`PaperAnalyzerAgent`)

**状态**：✅ 已完全实现（使用DeepSeek）

#### 功能概述
使用DeepSeek API深度分析清洗后的论文，提取核心内容和算法逻辑，输出结构化的Markdown分析报告。

#### 分析内容

**1. 论文核心内容**
- 主要研究问题
- 核心创新点
- 研究目标和动机

**2. 核心算法实现逻辑**
- 算法原理和理论基础
- 关键步骤和流程
- 技术细节和实现要点

**3. 技术亮点和贡献**
- 方法优势
- 实验结果
- 应用价值和影响

#### 工作流程

1. 从 `data/collections/all_papers_cleaned.json` 读取清洗后的论文
2. 逐篇调用DeepSeek API进行分析（约20-40秒/篇）
3. 保存Markdown格式的分析结果到 `data/analyzed/paper_*_analysis.md`
4. 创建分析集合 `data/collections/all_papers_analyzed.json`
5. 生成统计信息 `data/analyzed/analysis_stats.json`

#### 生成的文件

```
data/analyzed/
├── paper_1_analysis.md          # 论文1的Markdown分析
├── paper_2_analysis.md          # 论文2的Markdown分析
├── ...
└── analysis_stats.json          # 分析统计信息

data/collections/
└── all_papers_analyzed.json     # 分析结果集合
```

#### 输出格式示例

```markdown
## 论文核心内容

### 主要研究问题
本文研究了...

### 核心创新点
1. 首次提出...
2. 改进了...

## 核心算法实现逻辑

### 算法原理
基于...理论...

### 关键步骤
1. 构建模型
2. 优化求解
3. 验证结果

## 技术亮点和贡献

### 方法优势
- 计算效率高
- 精度提升显著

### 应用价值
可应用于...领域
```

#### 性能参数

- **分析时间**：20-40秒/篇
- **输出长度**：2000-2500字符/篇
- **API成本**：约$0.001-0.002/篇（DeepSeek）
- **成功率**：>95%

#### 使用示例

```python
from agents import PaperAnalyzerAgent

# 批量分析
agent = PaperAnalyzerAgent()
results = agent.run()  # 自动从清洗集合读取

# 单篇分析
analysis = agent.analyze_single("paper_1", paper_content)
print(analysis)  # Markdown格式的分析结果
```

#### UI操作

1. 点击 **"📖 论文处理"** Tab
2. 点击右侧 **"🔬 分析论文"** 按钮
3. 等待1-3分钟（取决于论文数量）
4. 使用 **"查看分析结果"** 查看Markdown格式的详细分析

---

### 4. 想法生成模块 (`IdeaGeneratorAgent`)

**状态**：✅ 已完全实现（使用DeepSeek）

#### 功能概述
基于多篇论文的分析结果，使用DeepSeek生成创新性强的研究想法，并对每个想法进行创新性评分。

#### 输入格式

从分析集合读取论文，按以下格式组织：
```
【Paper_1】论文名1：
分析内容1...

【Paper_2】论文名2：
分析内容2...

【Paper_3】论文名3：
分析内容3...
```

#### Prompt设计

```
这是我最近看的几篇文章，请尽量只根据这几篇文章的思路，
帮我想几个创新性比较强的idea(尽量详细一些)，
同时按照创新性对这几个idea进行打分。

要求：
1. 直接输出idea内容，不要开场白
2. 每个idea包含：标题、评分、详细描述
3. 使用Markdown格式
4. 按创新性从高到低排序
```

#### 工作流程

1. 从 `data/collections/all_papers_analyzed.json` 读取所有论文分析
2. 按【Paper_i】格式组织输入文本
3. 调用DeepSeek API生成创新想法（约1-2分钟）
4. 保存Markdown格式的想法到 `data/ideas/generated_ideas.md`

#### 生成的文件

```
data/ideas/
└── generated_ideas.md           # Markdown格式的创新想法（含评分）
```

#### 输出格式示例

```markdown
## Idea 1: 多模态张量分解的统一框架

**创新性评分**: 95/100

**核心思路**:
结合Paper_1的PARAFAC分解唯一性理论、Paper_2的MIMO雷达检测
技术和Paper_3的L型阵列处理方法，提出一个统一的多模态张量
分解框架...

**技术方案**:
1. 建立统一的张量模型
2. 设计自适应分解算法
3. 优化计算复杂度

**预期效果**:
- 提高DOA估计精度20%
- 降低计算复杂度30%
- 支持多种阵列配置

---

## Idea 2: 基于深度学习的张量分解加速

**创新性评分**: 88/100

...
```

#### 性能参数

- **生成时间**：1-2分钟（3篇论文）
- **输出长度**：3000-5000字符
- **API成本**：约$0.01-0.02（DeepSeek）
- **想法数量**：通常3-5个

#### 使用示例

```python
from agents import IdeaGeneratorAgent

# 自动从集合读取并生成
agent = IdeaGeneratorAgent()
ideas_text = agent.run()
print(ideas_text)

# 使用不同的API
agent = IdeaGeneratorAgent(api_provider="gemini", model="gemini-2.5-flash")
ideas_text = agent.run()
```

#### UI操作

1. 点击 **"💡 想法生成"** Tab
2. 点击 **"💡 生成想法"** 按钮
3. 等待1-2分钟
4. 使用 **"👁️ 查看想法"** 查看完整的Markdown格式想法

#### Prompt修改

**文件位置**: `agents/idea_generator_agent.py` (第174-186行)

可以修改Prompt来：
- 生成更多想法："请生成至少5个创新想法"
- 关注特定领域："请重点关注信号处理和机器学习的结合"
- 要求更详细："每个idea需要包含研究背景、核心创新点、技术实现方案、实验验证计划、预期贡献"

---

### 5. 想法筛选模块 (`IdeaSelectorAgent`)

**状态**：🔧 框架已实现，待完善

**计划功能**：
- 根据评分筛选最优想法
- 评估可行性
- 生成详细评估报告

---

### 6. 想法详细化模块 (`IdeaDetailerAgent`)

**状态**：🔧 框架已实现，待完善

**计划功能**：
- 将想法扩展为完整研究方案
- 包含方法论、实验设计
- 生成时间表

---

### 7. 代码生成模块 (`CodeGeneratorAgent`)

**状态**：🔧 框架已实现，待完善

**计划功能**：
- 根据想法生成Python实现
- 包含测试代码
- 包含文档注释

**输入**：`data/ideas/` 中的详细想法
**输出**：`data/code/` 中的Python代码

---

## 📄 MinerU PDF解析详细说明

### API集成实现

**MinerU Client** (`utils/mineru_client.py`)

实现了完整的MinerU API调用：

1. **文件上传API**：
   - `upload_file_and_extract()` - 单文件上传解析
   - `batch_upload_files_and_extract()` - 批量文件上传
   - 自动获取上传URL、执行PUT上传、创建解析任务

2. **URL解析API**：
   - `create_task()` - 创建单个URL解析任务
   - `batch_create_tasks()` - 创建批量URL解析任务

3. **任务管理**：
   - `get_task_status()` - 查询任务状态
   - `wait_for_task()` - 等待单个任务完成
   - `wait_for_batch()` - 等待批量任务完成

4. **结果处理**：
   - `download_result()` - 下载并解压结果
   - 自动保存Markdown、图片、JSON等

### 特点

- ✅ 高精度公式识别（LaTeX格式）
- ✅ 表格结构保持（Markdown表格）
- ✅ 图片自动提取（JPG/PNG格式）
- ✅ 支持批量处理（多文件并发）
- ✅ 支持文件上传（通过Web UI）
- ✅ 支持URL解析（公开链接）
- ✅ 每天2000页免费额度

### 模型选择

配置在 `config.yaml` 中：

```yaml
api:
  mineru:
    model_version: "vlm"  # 或 "pipeline"
```

- **VLM模型**（推荐）：基于视觉语言模型，精度高，适合复杂学术论文
- **Pipeline模型**：传统OCR流程，速度快，适合简单文档

### 测试MinerU

```bash
# 简单测试
python test_mineru_simple.py

# 完整测试（包括批量、文件上传）
python test_mineru.py
```

### 使用限制

- 单个文件最大100MB
- 每天2000页免费额度
- 解析时间依PDF复杂度：通常10-60秒
- 需要网络连接（调用云API）

---

## 📁 数据流和文件系统

### 完整数据流

```
用户输入 (PDF文件/URL)
    ↓
data/input/                      # 用户上传的PDF文件
    ├── paper1.pdf
    └── paper2.pdf
    ↓
[PDF提取模块]
    ↓
data/extracted/                  # 提取的文本和资源
    ├── paper1_extracted.txt     # 纯文本（用于后续处理）
    ├── paper1_mineru/           # MinerU完整结果
    │   ├── extracted/
    │   │   ├── full.md          # Markdown格式（含公式、表格）
    │   │   ├── layout.json      # 页面布局信息
    │   │   ├── {uuid}_content_list.json   # 内容列表
    │   │   ├── {uuid}_model.json          # 模型信息
    │   │   ├── {uuid}_origin.pdf          # 原始PDF
    │   │   └── images/          # 提取的所有图片
    │   │       ├── {hash1}.jpg  # 图片（hash命名）
    │   │       ├── {hash2}.jpg
    │   │       └── ...
    │   └── result.zip           # 原始ZIP（保留）
    ├── paper2_extracted.txt
    └── paper2_mineru/
    ↓
[论文清洗模块] (待实现)
    ↓
data/cleaned/                    # 清洗后的文本
    ├── paper1_cleaned.txt       # 移除参考文献、致谢等
    └── paper2_cleaned.txt
    ↓
[论文分析模块] (待实现)
    ↓
data/analyzed/                   # 分析结果
    ├── paper1_analysis.json     # 包含翻译、总结、公式分析
    │   {
    │     "title": "论文标题",
    │     "translation": "中文翻译",
    │     "summary": "核心总结",
    │     "methods": ["方法1", "方法2"],
    │     "formulas": [{"latex": "...", "explanation": "..."}],
    │     "contributions": ["贡献1", "贡献2"]
    │   }
    └── paper2_analysis.json
    ↓
[想法生成模块] (待实现)
    ↓
data/ideas/                      # 生成的想法
    ├── ideas_batch1.json        # 想法列表
    │   {
    │     "ideas": [
    │       {
    │         "id": "idea_001",
    │         "title": "想法标题",
    │         "description": "详细描述",
    │         "novelty_score": 85,
    │         "feasibility_score": 70,
    │         "impact_score": 90
    │       }
    │     ]
    │   }
    └── idea_001_detailed.json   # 详细化的想法
    ↓
[代码生成模块] (待实现)
    ↓
data/code/                       # 生成的代码
    ├── idea_001_implementation.py
    ├── idea_001_test.py
    └── idea_001_README.md
```

### 目录结构详解

```
Agent_Colab/
├── agents/                      # 所有Agent模块
│   ├── __init__.py
│   ├── base_agent.py            # 基类，提供日志、配置等
│   ├── pdf_extractor_agent.py   # ✅ PDF提取（已完成）
│   ├── paper_cleaner_agent.py   # 🔧 论文清洗（框架）
│   ├── paper_analyzer_agent.py  # 🔧 论文分析（框架）
│   ├── idea_generator_agent.py  # 🔧 想法生成（框架）
│   ├── idea_selector_agent.py   # 🔧 想法筛选（框架）
│   ├── idea_detailer_agent.py   # 🔧 想法详细化（框架）
│   └── code_generator_agent.py  # 🔧 代码生成（框架）
│
├── config/                      # 配置模块
│   ├── __init__.py
│   ├── api_config.py            # API配置定义
│   └── prompts.py               # 各模块的Prompt模板
│
├── utils/                       # 工具模块
│   ├── __init__.py
│   ├── api_client.py            # 统一API调用客户端
│   ├── mineru_client.py         # ✅ MinerU专用客户端（已完成）
│   ├── config_loader.py         # ✅ 配置加载器（已完成）
│   ├── file_manager.py          # 文件管理工具
│   └── logger.py                # ✅ 日志系统（已完成）
│
├── data/                        # 数据目录
│   ├── input/                   # 📥 输入：用户上传的PDF
│   ├── extracted/               # 📄 提取：文本+MinerU结果
│   ├── cleaned/                 # 🧹 清洗：移除无关内容
│   ├── analyzed/                # 🔍 分析：翻译+总结+公式
│   ├── ideas/                   # 💡 想法：生成的创新点
│   └── code/                    # 💻 代码：实现代码
│
├── logs/                        # 日志目录
│   └── agentcolab_YYYYMMDD.log  # 按日期的日志文件
│
├── config.yaml                  # ⚙️ 主配置文件（含API密钥）
├── config.example.yaml          # 📋 配置示例（不含密钥）
├── main.py                      # 🚀 命令行主程序
├── web_ui.py                    # 🎨 Web界面（Gradio）
├── run.sh                       # 🔧 启动脚本
├── requirements.txt             # 📦 Python依赖
├── test_setup.py                # 🧪 环境测试
├── test_mineru.py               # 🧪 MinerU完整测试
├── test_mineru_simple.py        # 🧪 MinerU简单测试
├── README.md                    # 📖 本文档
├── ENV_SETUP.md                 # 🔑 环境配置指南
└── .gitignore                   # 🔒 Git忽略规则
```

### 文件命名规则

**提取文件**：
- `{论文名}_extracted.txt` - 纯文本
- `{论文名}_mineru/` - MinerU完整结果目录

**清洗文件**：
- `{论文名}_cleaned.txt` - 清洗后文本

**分析文件**：
- `{论文名}_analysis.json` - JSON格式分析结果

**想法文件**：
- `ideas_batch{N}.json` - 想法列表
- `idea_{ID}_detailed.json` - 详细化想法

**代码文件**：
- `idea_{ID}_implementation.py` - 实现代码
- `idea_{ID}_test.py` - 测试代码
- `idea_{ID}_README.md` - 说明文档

### 日志文件

**位置**：`logs/agentcolab_YYYYMMDD.log`

**内容示例**：
```
2024-01-15 10:30:45 - AgentColab - INFO - [PDF提取Agent] 开始任务
2024-01-15 10:30:46 - AgentColab - INFO - 使用MinerU提取: paper1.pdf
2024-01-15 10:30:47 - AgentColab - INFO - 创建MinerU任务: task_abc123
2024-01-15 10:31:02 - AgentColab - INFO - 任务状态: running
2024-01-15 10:31:17 - AgentColab - INFO - 任务状态: done
2024-01-15 10:31:18 - AgentColab - INFO - 下载结果到: data/extracted/paper1_mineru
2024-01-15 10:31:20 - AgentColab - INFO - ✓ 提取成功，保存到: data/extracted/paper1_extracted.txt
```

**日志级别配置**（`config.yaml`）：
```yaml
logging:
  level: "INFO"  # DEBUG | INFO | WARNING | ERROR | CRITICAL
```

---

## 🧪 测试和验证

### 环境测试

```bash
# 完整环境测试
python test_setup.py

# MinerU测试
python test_mineru_simple.py

# 或使用启动脚本
./run.sh check
```

### 测试单个模块

```python
# 测试PDF提取
from agents import PDFExtractorAgent
agent = PDFExtractorAgent(use_mineru=False)
results = agent.run()

# 测试论文分析
from agents import PaperAnalyzerAgent
agent = PaperAnalyzerAgent()
results = agent.run()
```

---

## ❓ 常见问题

### Q1: 如何设置API密钥？

**A**: 推荐使用环境变量（优先级最高）：

```bash
# 临时设置（当前终端会话）
export GOOGLE_API_KEY="your_gemini_key"
export DEEPSEEK_API_KEY="your_deepseek_key"
export ANTHROPIC_API_KEY="your_claude_key"
export MINERU_API_KEY="your_mineru_key"

# 永久设置（添加到 ~/.bashrc 或 ~/.zshrc）
echo 'export GOOGLE_API_KEY="your_key"' >> ~/.bashrc
source ~/.bashrc

# 或使用 .env 文件
cat > .env << EOF
GOOGLE_API_KEY=your_key
DEEPSEEK_API_KEY=your_key
ANTHROPIC_API_KEY=your_key
MINERU_API_KEY=your_key
EOF

# 或在 config.yaml 中配置（优先级较低）
# 编辑 config.yaml 的 api_keys 部分
```

**验证配置**：
```bash
./run.sh check
# 或
python test_setup.py
```

---

### Q2: MinerU和PyPDF2如何选择？

**A**: 根据需求选择：

| 特性 | MinerU | PyPDF2 |
|------|--------|--------|
| **精度** | ⭐⭐⭐⭐⭐ 高 | ⭐⭐ 低 |
| **公式识别** | ✅ 支持（LaTeX） | ❌ 不支持 |
| **表格识别** | ✅ 支持（Markdown） | ❌ 不支持 |
| **图片提取** | ✅ 自动提取 | ❌ 不支持 |
| **输入方式** | URL + 文件上传 | 仅本地文件 |
| **网络要求** | ✅ 需要 | ❌ 不需要 |
| **API密钥** | ✅ 需要 | ❌ 不需要 |
| **免费额度** | 2000页/天 | ♾️ 无限 |
| **速度** | 10-60秒/文档 | <1秒/文档 |
| **适用场景** | 学术论文、复杂文档 | 简单文本文档 |

**推荐**：
- 📚 学术论文（含公式、表格）→ 使用 MinerU
- 📄 简单文档（纯文本）→ 使用 PyPDF2
- 🚫 无网络环境 → 使用 PyPDF2

**切换方式**：
```python
# 使用MinerU
agent = PDFExtractorAgent(use_mineru=True)

# 使用PyPDF2
agent = PDFExtractorAgent(use_mineru=False)
```

---

### Q3: 如何处理本地PDF文件？

**A**: 三种方式：

**方式1：Web UI上传（推荐）**
```bash
./run.sh ui
# 在浏览器中：PDF提取 → 上传文件 → 选择MinerU → 开始提取
```

**方式2：Python代码上传到MinerU**
```python
from agents import PDFExtractorAgent

agent = PDFExtractorAgent(use_mineru=True)
content = agent.extract_from_file(
    pdf_path="path/to/paper.pdf",
    pdf_name="my_paper"
)
# 自动上传到MinerU服务器并解析
```

**方式3：使用PyPDF2本地处理**
```python
# 1. 将PDF放入data/input/目录
# 2. 使用PyPDF2提取
agent = PDFExtractorAgent(use_mineru=False)
results = agent.run()
```

---

### Q4: MinerU提取的文件在哪里？

**A**: 完整结构如下：

```
data/extracted/
├── 论文名_extracted.txt          # 纯文本（用于后续Agent处理）
└── 论文名_mineru/                 # MinerU完整结果
    ├── extracted/
    │   ├── full.md                # Markdown格式（含公式、表格）
    │   ├── layout.json            # 布局信息
    │   ├── {uuid}_content_list.json   # 内容列表
    │   ├── {uuid}_model.json          # 模型信息
    │   ├── {uuid}_origin.pdf          # 原始PDF
    │   └── images/                # 所有提取的图片
    │       ├── {hash1}.jpg
    │       ├── {hash2}.jpg
    │       └── ...
    └── result.zip                 # 原始ZIP文件（保留）
```

**访问方式**：
```python
import os
import json

# 读取纯文本（用于后续Agent处理）
with open('data/extracted/论文名_extracted.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# 读取Markdown（含公式和表格）
with open('data/extracted/论文名_mineru/extracted/full.md', 'r', encoding='utf-8') as f:
    markdown = f.read()

# 读取布局信息
with open('data/extracted/论文名_mineru/extracted/layout.json', 'r', encoding='utf-8') as f:
    layout = json.load(f)

# 读取内容列表（找到UUID前缀的文件）
extracted_dir = 'data/extracted/论文名_mineru/extracted/'
content_list_file = [f for f in os.listdir(extracted_dir) if f.endswith('_content_list.json')][0]
with open(os.path.join(extracted_dir, content_list_file), 'r', encoding='utf-8') as f:
    content_list = json.load(f)

# 查看提取的图片
images_dir = 'data/extracted/论文名_mineru/extracted/images/'
images = os.listdir(images_dir)
print(f"提取了 {len(images)} 张图片")
```

---

### Q5: 完整流程需要多长时间？

**A**: 取决于多个因素：

| 阶段 | 单篇论文 | 3篇论文 | 10篇论文 |
|------|---------|---------|----------|
| **PDF提取（MinerU）** | 10-60秒 | 30-180秒 | 5-10分钟 |
| **PDF提取（PyPDF2）** | <1秒 | 1-3秒 | 3-10秒 |
| **论文清洗** | 待实现 | 待实现 | 待实现 |
| **论文分析** | 待实现 | 待实现 | 待实现 |
| **想法生成** | 待实现 | 待实现 | 待实现 |
| **代码生成** | 待实现 | 待实现 | 待实现 |

**影响因素**：
- PDF复杂度（页数、公式、表格数量）
- 网络速度
- API响应时间
- 服务器负载

**建议**：
- 首次使用建议单篇测试
- 批量处理建议5篇以下
- 可在Web UI中实时查看进度

---

### Q6: API调用失败怎么办？

**A**: 系统化排查：

**1. 检查API密钥**
```bash
# 查看环境变量
echo $GOOGLE_API_KEY
echo $MINERU_API_KEY

# 检查配置
cat config.yaml | grep api_key

# 验证密钥
python test_setup.py
```

**2. 检查网络连接**
```bash
# 测试Gemini连接
curl -H "x-goog-api-key: $GOOGLE_API_KEY" \
  https://generativelanguage.googleapis.com/v1beta/models

# 测试MinerU连接
curl https://mineru.net/api/v4/extract/task
```

**3. 查看日志**
```bash
# 查看最新日志
tail -f logs/agentcolab_*.log

# 搜索错误
grep -i error logs/agentcolab_*.log
```

**4. 检查API额度**
- Gemini: https://aistudio.google.com/app/apikey
- MinerU: 登录 https://mineru.net 查看剩余额度

**常见错误**：
```
错误: "API key not valid"
解决: 检查密钥是否正确，注意前后空格

错误: "Rate limit exceeded"
解决: 等待一段时间或升级API套餐

错误: "Connection timeout"
解决: 检查网络，或增加timeout配置

错误: "Task failed: xxx"
解决: 检查PDF文件是否损坏或格式不支持
```

---

### Q7: 如何查看处理结果？

**A**: 多种方式：

**方式1：文件系统**
```bash
# 查看提取结果
ls -lh data/extracted/
cat data/extracted/论文名_extracted.txt

# 查看MinerU图片
ls -lh data/extracted/论文名_mineru/extracted/images/

# 查看其他结果
ls -lh data/cleaned/
ls -lh data/analyzed/
ls -lh data/ideas/
ls -lh data/code/
```

**方式2：Web UI**
- 每个模块运行后会显示结果
- 可直接下载文件

**方式3：Python代码**
```python
import json

# 读取分析结果
with open('data/analyzed/paper1_analysis.json', 'r') as f:
    analysis = json.load(f)
    print(analysis['summary'])

# 读取想法
with open('data/ideas/ideas_batch1.json', 'r') as f:
    ideas = json.load(f)
    for idea in ideas['ideas']:
        print(f"{idea['title']}: {idea['novelty_score']}")
```

---

### Q8: Web UI无法启动？

**A**: 逐步排查：

**检查依赖**：
```bash
pip install gradio>=4.0.0
pip install -r requirements.txt
```

**检查端口占用**：
```bash
# 查看7860端口是否被占用
lsof -i :7860

# 如被占用，杀掉进程
kill -9 <PID>

# 或使用其他端口
python web_ui.py --server-port 7861
```

**查看错误信息**：
```bash
# 直接运行查看详细错误
python web_ui.py

# 检查Python版本（需要3.8+）
python --version
```

**常见问题**：
```
错误: "ModuleNotFoundError: No module named 'gradio'"
解决: pip install gradio

错误: "Address already in use"
解决: 更换端口或结束占用进程

错误: "TypeError: BlockContext.__init__() got an unexpected keyword argument 'theme'"
解决: 升级Gradio: pip install --upgrade gradio
```

---

### Q9: 如何自定义Prompt？

**A**: 两种方式：

**方式1：修改prompts.py**
```python
# 编辑 config/prompts.py
PAPER_TRANSLATION_PROMPT = """
你的自定义翻译prompt...
"""

PAPER_SUMMARY_PROMPT = """
你的自定义总结prompt...
"""
```

**方式2：在config.yaml中覆盖**
```yaml
prompts:
  paper_translation: |
    你的自定义翻译prompt...
    可以多行...
  
  paper_summary: |
    你的自定义总结prompt...
```

---

### Q10: 如何批量处理大量PDF？

**A**: 建议策略：

**小批量处理**：
```python
from agents import PDFExtractorAgent

agent = PDFExtractorAgent(use_mineru=True)

# 分批处理，每批5个
batch_size = 5
all_pdfs = ["pdf1.pdf", "pdf2.pdf", ..., "pdf100.pdf"]

for i in range(0, len(all_pdfs), batch_size):
    batch = all_pdfs[i:i+batch_size]
    results = agent.batch_extract_from_files(batch)
    print(f"完成第 {i//batch_size + 1} 批")
    time.sleep(60)  # 避免API限流
```

**注意事项**：
- MinerU每天2000页限额
- 建议每批5个以下
- 批次间隔60秒以上
- 监控日志文件

---

## 📊 依赖包

主要依赖：

```
google-generativeai  # Gemini API
anthropic            # Claude API
openai               # DeepSeek API
gradio               # Web UI
PyPDF2               # PDF处理
pyyaml               # 配置解析
requests             # HTTP请求
```

安装：

```bash
pip install -r requirements.txt
```

---

## 🔒 安全最佳实践

1. **不要将API密钥提交到Git**
   - `.gitignore` 已配置保护
   - 使用环境变量而非配置文件

2. **使用环境变量**
   ```bash
   export GOOGLE_API_KEY="your_key"
   ```

3. **定期轮换密钥**
   - 定期更换API密钥
   - 区分开发和生产环境

4. **限制权限**
   - 只授予必要的API权限
   - 监控API使用量

---

## 🛠️ 开发和扩展

### 添加新的Agent

```python
from agents.base_agent import BaseAgent

class MyCustomAgent(BaseAgent):
    def __init__(self):
        super().__init__("自定义Agent")
    
    def run(self, *args, **kwargs):
        self.log_start("开始任务")
        # 你的逻辑
        self.log_end("任务完成")
        return result
```

### 自定义流程

```python
from agents import PDFExtractorAgent, PaperAnalyzerAgent

# 自定义工作流
pdf_agent = PDFExtractorAgent(use_mineru=True)
papers = pdf_agent.run()

# 只分析前3篇
analyzer = PaperAnalyzerAgent()
selected = {k: v for k, v in list(papers.items())[:3]}
results = analyzer.run(selected)
```

---

## 📝 更新日志

### 当前版本特性

- ✅ 完整的PDF处理流程
- ✅ MinerU高精度PDF解析
- ✅ 多种API集成（Gemini、DeepSeek、Claude）
- ✅ Web UI界面
- ✅ 批量处理支持
- ✅ 灵活的配置系统
- ✅ 完整的日志记录

### 计划功能

- ⏳ 实验结果分析
- ⏳ 自动生成论文
- ⏳ 更多PDF解析引擎
- ⏳ 分布式处理支持

---

## 🔧 UI功能调用关系与Prompt修改

### UI界面功能映射

#### Tab 1: 配置管理
- **函数**: `get_current_config()`, `save_api_keys()`
- **文件**: `web_ui.py` (第28-50行)
- **作用**: 读取和保存API密钥

#### Tab 2: PDF提取
- **上传文件**: `extract_pdf_from_upload()` → `PDFExtractorAgent` → `mineru_client.upload_file_and_extract()` 或 `PyPDF2`
- **URL提取**: `extract_pdf_from_url()` → `mineru_client.extract_pdf_from_url()`
- **批量处理**: `batch_extract_pdfs_upload()`, `batch_extract_pdfs_url()`
- **保存**: `data/extracted/`, `data/collections/all_papers.json`

#### Tab 2.5: 论文集合
- **查看集合**: `load_collection_info()` → `PaperCollection.load_from_json()`
- **创建集合**: `PaperCollection.from_extracted_dir()`
- **文件**: `utils/collection_ui.py`, `utils/paper_collection.py`

#### Tab 3: 论文处理

**清洗论文** (无Prompt，纯Python规则)
- **函数**: `clean_papers()` → `PaperCleanerAgent.run()`
- **文件**: `agents/paper_cleaner_agent.py`
- **输入**: `data/collections/all_papers.json`
- **输出**: `data/cleaned/`, `data/collections/all_papers_cleaned.json`

**分析论文** (使用DeepSeek + Prompt)
- **函数**: `analyze_papers()` → `PaperAnalyzerAgent.run()`
- **文件**: `agents/paper_analyzer_agent.py`
- **输入**: `data/collections/all_papers_cleaned.json`
- **输出**: `data/analyzed/paper_*_analysis.md`, `data/collections/all_papers_analyzed.json`

### Prompt修改指南

#### 当前Prompt位置

**论文分析Prompt**:
```python
# 文件: agents/paper_analyzer_agent.py (第22-33行)
self.analysis_prompt = """请总结一下这篇文章的核心，以及核心算法实现逻辑。

要求：
1. 请用中文回答
2. 使用Markdown格式组织内容
3. 包含以下部分：
   - 论文核心内容（主要研究问题、创新点）
   - 核心算法实现逻辑（算法原理、关键步骤）
   - 技术亮点和贡献

论文内容：
{paper_content}"""
```

#### 如何修改Prompt

**方法1: 直接修改代码**
```python
# 编辑 agents/paper_analyzer_agent.py
self.analysis_prompt = """你的新Prompt内容

要求：
1. ...
2. ...

论文内容：
{paper_content}"""  # ⚠️ 必须保留 {paper_content}
```

**方法2: 修改System Prompt**
```python
# 在 _analyze_paper() 方法中 (第95行)
analysis = self.deepseek_client.generate(
    prompt=prompt,
    system_prompt="你是一位资深的学术论文分析专家..."  # 修改这里
)
```

#### Prompt修改示例

**示例1: 更详细的分析**
```python
self.analysis_prompt = """请对这篇论文进行深度分析。

分析要求：
1. **研究背景**: 说明研究领域和现有问题
2. **核心创新**: 详细说明本文的创新点
3. **方法论**: 
   - 理论基础
   - 算法设计
   - 实现步骤
4. **实验验证**:
   - 实验设置
   - 对比方法
   - 性能指标
5. **应用价值**: 实际应用场景和影响

输出格式: Markdown，使用中文

论文内容：
{paper_content}"""
```

**示例2: 针对特定领域**
```python
self.analysis_prompt = """请从信号处理角度分析这篇论文。

重点关注：
1. 信号模型的建立
2. 算法的计算复杂度
3. 对噪声的鲁棒性
4. 与经典方法的对比

论文内容：
{paper_content}"""
```

#### Prompt调试技巧

1. **查看实际发送的Prompt**
   ```python
   # 在 _analyze_paper() 中添加
   prompt = self.analysis_prompt.format(paper_content=paper_content)
   self.logger.info(f"发送的Prompt: {prompt[:500]}...")
   ```

2. **测试不同的Prompt**
   ```bash
   # 修改后在UI测试，查看结果
   cat data/analyzed/paper_1_analysis.md
   ```

3. **控制输出长度**
   - 在Prompt中添加: "请将分析控制在2000字以内"
   - 或修改 `utils/api_client.py` 中的 `max_tokens`

4. **使用Few-Shot示例**
   ```python
   self.analysis_prompt = """
   示例输入：
   论文标题: XXX
   论文内容: ...
   
   示例输出：
   ## 核心内容
   本文研究了...
   
   现在请分析以下论文：
   {paper_content}
   """
   ```

#### 相关文件速查

- **Prompt定义**: `agents/paper_analyzer_agent.py` (第22-33行)
- **API调用**: `utils/api_client.py` (`DeepSeekClient.generate`, 第119行)
- **API配置**: `config.yaml` (deepseek部分), `utils/config_loader.py`
- **UI触发**: `web_ui.py` (`analyze_papers`, 第325行)

---

## 📧 支持和反馈

- 查看日志：`logs/agentcolab_*.log`
- 环境检查：`./run.sh check`
- 所有功能通过UI测试：`./run.sh ui`

---

## 📄 License

MIT License

---

**AgentColab - 让论文处理和创新研究更简单！** 🎉

**快速开始**：`./run.sh ui`
