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
   - ✅ 使用DeepSeek API自动清理附录、参考文献等

3. **深度论文分析**
   - ✅ 使用Gemini API翻译成中文
   - ✅ 分析和推导公式
   - ✅ 总结核心算法

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

```python
from agents import PDFExtractorAgent, PaperAnalyzerAgent

# 使用MinerU提取PDF
agent = PDFExtractorAgent(use_mineru=True)
content = agent.extract_from_url("https://example.com/paper.pdf")

# 分析论文
analyzer = PaperAnalyzerAgent()
results = analyzer.run()

# 或使用主程序
from main import AgentColab
autopaper = AgentColab()
results = autopaper.run_full_pipeline()
```

---

## ⚙️ 配置说明

### API密钥配置

**优先级**：环境变量 > 配置文件 > 空

配置文件位置：`config.yaml`

```yaml
# API密钥配置
api_keys:
  google_api_key: ""      # Gemini API
  deepseek_api_key: ""    # DeepSeek API
  anthropic_api_key: ""   # Claude API
  mineru_api_key: ""      # MinerU API（可选）

# API参数配置
api:
  gemini:
    model: "gemini-2.5-flash"
    temperature: 0.7
    max_output_tokens: 8192
  
  deepseek:
    base_url: "https://api.deepseek.com"
    model: "deepseek-chat"
    temperature: 0.7
  
  claude:
    model: "claude-3-5-sonnet-20241022"
    temperature: 0.7
  
  mineru:
    base_url: "https://mineru.net/api/v4"
    model_version: "vlm"
    enable_formula: true
    enable_table: true
```

### 流程控制配置

```yaml
pipeline:
  # PDF提取配置
  pdf_extraction:
    use_mineru: false           # 是否使用MinerU
    fallback_to_pypdf2: true    # 回退到PyPDF2
    mineru_model: "vlm"         # VLM或Pipeline
  
  # 论文清洗配置
  paper_cleaning:
    enabled: true
  
  # 论文分析配置
  paper_analysis:
    do_translation: true        # 翻译
    do_summary: true            # 总结
  
  # 想法生成配置
  idea_generation:
    min_ideas: 3
    score_threshold: 60
```

---

## 📄 MinerU PDF解析

MinerU是推荐的PDF解析方案，提供高精度提取。

### 特点

- ✅ 高精度公式识别（LaTeX格式）
- ✅ 表格结构保持
- ✅ 图片自动提取
- ✅ 支持批量处理
- ✅ 每天2000页免费额度

### 使用方法

**单个PDF（从URL）**：

```python
from agents import PDFExtractorAgent

agent = PDFExtractorAgent(use_mineru=True)
content = agent.extract_from_url(
    pdf_url="https://example.com/paper.pdf",
    pdf_name="my_paper",
    model_version="vlm"  # 推荐VLM模型
)
```

**批量处理**：

```python
pdf_urls = [
    "https://example.com/paper1.pdf",
    "https://example.com/paper2.pdf"
]

results = agent.extract_from_urls(
    pdf_urls=pdf_urls,
    pdf_names=["paper1", "paper2"]
)
```

### 注意事项

⚠️ **MinerU需要PDF的公开URL**，不支持本地文件直接上传

**解决方案**：
- 将PDF上传到云存储（阿里云OSS、腾讯云COS等）获取URL
- 或使用PyPDF2处理本地文件：`PDFExtractorAgent(use_mineru=False)`

### 模型选择

- **VLM模型**（推荐）：基于视觉语言模型，精度高，适合复杂学术论文
- **Pipeline模型**：传统OCR流程，速度快，适合简单文档

### 测试MinerU

```bash
python test_mineru_simple.py
```

---

## 📁 项目结构

```
Agent_Colab/
├── agents/              # 各功能Agent
│   ├── base_agent.py
│   ├── pdf_extractor_agent.py
│   ├── paper_cleaner_agent.py
│   ├── paper_analyzer_agent.py
│   ├── idea_generator_agent.py
│   ├── idea_selector_agent.py
│   ├── idea_detailer_agent.py
│   └── code_generator_agent.py
├── config/              # 配置模块
│   ├── api_config.py
│   └── prompts.py
├── utils/               # 工具模块
│   ├── api_client.py
│   ├── mineru_client.py
│   ├── config_loader.py
│   ├── file_manager.py
│   └── logger.py
├── data/                # 数据目录
│   ├── input/           # PDF输入
│   ├── extracted/       # 提取结果
│   ├── cleaned/         # 清洗结果
│   ├── analyzed/        # 分析结果
│   ├── ideas/           # 生成的想法
│   └── code/            # 生成的代码
├── logs/                # 日志目录
├── config.yaml          # 主配置文件
├── main.py              # 主程序
├── web_ui.py            # Web界面
├── run.sh               # 启动脚本
├── start_ui.sh          # UI启动脚本
└── requirements.txt     # 依赖包
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

**A**: 三种方式，推荐使用环境变量：

```bash
# 临时设置（当前会话）
export GOOGLE_API_KEY="your_key"

# 永久设置（添加到 ~/.bashrc）
echo 'export GOOGLE_API_KEY="your_key"' >> ~/.bashrc
source ~/.bashrc

# 或在 config.yaml 中配置
```

### Q2: MinerU和PyPDF2如何选择？

**A**: 
- **MinerU**：精度高，支持公式表格，需要PDF URL，每天2000页免费
- **PyPDF2**：速度快，可处理本地文件，精度较低，完全免费

推荐：学术论文用MinerU，简单文档用PyPDF2

### Q3: 如何处理本地PDF文件？

**A**: 两种方案：

```bash
# 方案1: 上传到云存储获取URL，使用MinerU
agent = PDFExtractorAgent(use_mineru=True)
agent.extract_from_url("https://your-storage.com/paper.pdf")

# 方案2: 直接使用PyPDF2
agent = PDFExtractorAgent(use_mineru=False)
agent.run()  # 自动处理 data/input/ 目录
```

### Q4: 完整流程需要多长时间？

**A**: 取决于论文数量和API速度：
- 单篇论文：约5-10分钟
- 3-5篇论文：约20-30分钟
- 建议先小批量测试

### Q5: API调用失败怎么办？

**A**: 检查步骤：
1. 确认API密钥正确（无多余空格）
2. 检查网络连接
3. 查看日志文件：`logs/agentcolab_*.log`
4. 验证API额度是否充足

### Q6: 如何查看处理结果？

**A**: 所有结果保存在 `data/` 目录：

```bash
ls -la data/extracted/   # 提取的文本
ls -la data/cleaned/     # 清洗后的文本
ls -la data/analyzed/    # 分析结果
ls -la data/ideas/       # 生成的想法
ls -la data/code/        # 生成的代码
```

### Q7: Web UI无法启动？

**A**: 
```bash
# 检查Gradio是否安装
pip install gradio>=4.0.0

# 检查端口是否被占用
lsof -i :7860

# 查看错误日志
python web_ui.py
```

### Q8: 如何自定义Prompt？

**A**: 编辑 `config/prompts.py` 或在 `config.yaml` 中覆盖：

```yaml
prompts:
  paper_translation: "你的自定义prompt"
  paper_summary: "你的自定义prompt"
```

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

## 📧 支持和反馈

- 查看日志：`logs/agentcolab_*.log`
- 环境检查：`./run.sh check`
- 测试脚本：`python test_setup.py`

---

## 📄 License

MIT License

---

**AgentColab - 让论文处理和创新研究更简单！** 🎉

**快速开始**：`./run.sh ui`
