#!/bin/bash

# AgentColab 启动脚本
# 使用方法: ./run.sh [命令] [选项]

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 项目根目录
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

# 打印带颜色的消息
print_info() {
    echo -e "${BLUE}[信息]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[成功]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[警告]${NC} $1"
}

print_error() {
    echo -e "${RED}[错误]${NC} $1"
}

# 打印Logo
print_logo() {
    echo -e "${BLUE}"
    cat << "EOF"
╔════════════════════════════════════════════════════════════╗
║                      AgentColab 系统                        ║
║          自动论文处理与创新想法生成系统                      ║
╚════════════════════════════════════════════════════════════╝
EOF
    echo -e "${NC}"
}

# 检查Python环境
check_python() {
    print_info "检查Python环境..."
    
    if ! command -v python3 &> /dev/null; then
        print_error "未找到Python3，请先安装Python3"
        exit 1
    fi
    
    PYTHON_VERSION=$(python3 --version | awk '{print $2}')
    print_success "Python版本: $PYTHON_VERSION"
}

# 检查并创建虚拟环境
check_venv() {
    if [ ! -d "venv" ]; then
        print_warning "未找到虚拟环境，正在创建..."
        python3 -m venv venv
        print_success "虚拟环境创建完成"
    fi
}

# 激活虚拟环境
activate_venv() {
    print_info "激活虚拟环境..."
    source venv/bin/activate
    print_success "虚拟环境已激活"
}

# 安装依赖
install_dependencies() {
    print_info "安装项目依赖..."
    pip install -r requirements.txt -q
    print_success "依赖安装完成"
}

# 检查API密钥
check_api_keys() {
    print_info "检查API密钥配置..."
    
    local all_set=true
    
    if [ -z "$GOOGLE_API_KEY" ]; then
        print_warning "GOOGLE_API_KEY 未设置"
        all_set=false
    else
        print_success "GOOGLE_API_KEY 已设置"
    fi
    
    if [ -z "$DEEPSEEK_API_KEY" ]; then
        print_warning "DEEPSEEK_API_KEY 未设置"
        all_set=false
    else
        print_success "DEEPSEEK_API_KEY 已设置"
    fi
    
    if [ -z "$ANTHROPIC_API_KEY" ]; then
        print_warning "ANTHROPIC_API_KEY 未设置"
        all_set=false
    else
        print_success "ANTHROPIC_API_KEY 已设置"
    fi
    
    if [ -z "$MINERU_API_KEY" ]; then
        print_info "MINERU_API_KEY 未设置（可选）"
    else
        print_success "MINERU_API_KEY 已设置"
    fi
    
    if [ "$all_set" = false ]; then
        echo ""
        print_warning "部分必需的API密钥未设置"
        print_info "请参考 env_template.txt 设置环境变量"
        echo ""
    fi
}

# 检查目录结构
check_directories() {
    print_info "检查目录结构..."
    
    directories=("data/input" "data/extracted" "data/cleaned" "data/analyzed" "data/ideas" "data/code" "logs")
    
    for dir in "${directories[@]}"; do
        if [ ! -d "$dir" ]; then
            mkdir -p "$dir"
            print_info "创建目录: $dir"
        fi
    done
    
    print_success "目录结构检查完成"
}

# 显示帮助信息
show_help() {
    cat << EOF
AgentColab 启动脚本使用说明

用法:
    ./run.sh [命令] [选项]

命令列表:
    setup       - 初始化项目环境（创建虚拟环境、安装依赖）
    ui          - 启动Web UI界面
    full        - 运行完整流程（从PDF到代码生成）
    pdf         - 仅运行PDF提取模块
    clean       - 仅运行论文清洗模块
    analyze     - 仅运行论文分析模块
    idea        - 仅运行想法生成模块
    select      - 仅运行想法筛选模块
    detail      - 仅运行想法详细化模块
    code        - 仅运行代码生成模块
    check       - 检查环境配置
    help        - 显示此帮助信息

选项:
    --no-venv   - 不使用虚拟环境

示例:
    ./run.sh setup              # 初始化项目
    ./run.sh ui                 # 启动Web UI（推荐）
    ./run.sh full               # 运行完整流程
    ./run.sh pdf                # 只提取PDF
    ./run.sh check              # 检查环境配置
    ./run.sh full --no-venv     # 不使用虚拟环境运行

注意事项:
    1. 首次使用请先运行: ./run.sh setup
    2. 请先将PDF文件放入 data/input/ 目录
    3. 确保已设置必要的API密钥环境变量
       可以参考 env_template.txt 文件

EOF
}

# 初始化项目
setup_project() {
    print_logo
    print_info "开始初始化AgentColab项目..."
    echo ""
    
    check_python
    check_venv
    activate_venv
    install_dependencies
    check_directories
    
    echo ""
    print_success "项目初始化完成！"
    echo ""
    print_info "下一步："
    echo "  1. 设置API密钥（参考 env_template.txt）"
    echo "  2. 将PDF文件放入 data/input/ 目录"
    echo "  3. 运行: ./run.sh full"
    echo ""
}

# 运行主程序
run_main() {
    local command=$1
    local use_venv=true
    
    # 检查是否使用 --no-venv 选项
    if [ "$2" = "--no-venv" ]; then
        use_venv=false
    fi
    
    print_logo
    
    # 使用虚拟环境
    if [ "$use_venv" = true ]; then
        if [ ! -d "venv" ]; then
            print_error "虚拟环境不存在，请先运行: ./run.sh setup"
            exit 1
        fi
        activate_venv
    fi
    
    check_api_keys
    check_directories
    
    echo ""
    print_info "开始执行: $command"
    echo ""
    
    # 运行Python程序
    python3 main.py "$command"
    
    # 检查执行结果
    if [ $? -eq 0 ]; then
        echo ""
        print_success "执行完成！"
    else
        echo ""
        print_error "执行失败，请查看日志文件"
    fi
}

# 检查环境
check_environment() {
    print_logo
    print_info "检查环境配置..."
    echo ""
    
    check_python
    
    if [ -d "venv" ]; then
        print_success "虚拟环境: 已创建"
    else
        print_warning "虚拟环境: 未创建"
    fi
    
    echo ""
    check_api_keys
    
    echo ""
    check_directories
    
    # 检查配置文件
    echo ""
    print_info "检查配置文件..."
    if [ -f "config.yaml" ]; then
        print_success "config.yaml 存在"
    else
        print_warning "config.yaml 不存在"
    fi
    
    # 检查PDF文件
    echo ""
    print_info "检查输入文件..."
    pdf_count=$(find data/input -name "*.pdf" 2>/dev/null | wc -l)
    if [ $pdf_count -gt 0 ]; then
        print_success "在 data/input 中找到 $pdf_count 个PDF文件"
    else
        print_warning "在 data/input 中未找到PDF文件"
    fi
    
    echo ""
}

# 启动Web UI
start_ui() {
    print_logo
    print_info "启动Web UI..."
    echo ""
    
    # 使用虚拟环境
    if [ ! -d "venv" ]; then
        print_error "虚拟环境不存在，请先运行: ./run.sh setup"
        exit 1
    fi
    activate_venv
    
    # 检查并安装依赖
    print_info "检查依赖包..."
    if ! python3 -c "import google.generativeai" 2>/dev/null; then
        print_info "安装项目依赖（首次运行需要一些时间）..."
        pip install -r requirements.txt -q
        print_success "依赖安装完成"
    fi
    
    print_success "Web UI启动中..."
    echo ""
    print_info "界面将在浏览器中打开"
    print_info "访问地址: http://localhost:7860"
    echo ""
    print_info "按 Ctrl+C 停止服务器"
    echo ""
    
    # 运行UI
    python3 web_ui.py
}

# 主逻辑
main() {
    # 给脚本添加执行权限（如果没有的话）
    if [ ! -x "$0" ]; then
        chmod +x "$0"
    fi
    
    case "${1:-help}" in
        setup)
            setup_project
            ;;
        ui)
            start_ui
            ;;
        full|pdf|clean|analyze|idea|select|detail|code)
            run_main "$@"
            ;;
        check)
            check_environment
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            print_error "未知的命令: $1"
            echo ""
            show_help
            exit 1
            ;;
    esac
}

# 执行主函数
main "$@"

