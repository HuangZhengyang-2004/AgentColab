"""
PDF提取Agent
负责调用MinerU API从PDF文件中提取文本内容
"""

import os
from pathlib import Path
from typing import List, Dict, Optional
import PyPDF2

from agents.base_agent import BaseAgent
from utils.mineru_client import get_mineru_client


class PDFExtractorAgent(BaseAgent):
    """PDF文档提取Agent"""
    
    def __init__(self, use_mineru: bool = True):
        """
        初始化PDF提取Agent
        
        Args:
            use_mineru: 是否优先使用MinerU，False则直接使用PyPDF2
        """
        super().__init__("PDF提取Agent")
        
        self.use_mineru = use_mineru
        self.mineru_client = None
        
        if use_mineru:
            try:
                # 优先从环境变量读取，其次从配置文件读取
                api_key = os.getenv('MINERU_API_KEY', '')
                
                if not api_key:
                    # 尝试从配置文件读取
                    try:
                        from utils.config_loader import config_loader
                        config = config_loader.get_api_config('mineru')
                        api_key = config.get('api_key', '')
                    except Exception as e:
                        self.logger.debug(f"从配置文件读取密钥失败: {e}")
                
                if api_key:
                    self.mineru_client = get_mineru_client(api_key)
                    self.logger.info("✓ MinerU客户端初始化成功")
                else:
                    self.logger.warning("未设置MINERU_API_KEY，将使用PyPDF2")
            except Exception as e:
                self.logger.warning(f"MinerU客户端初始化失败，将使用PyPDF2: {str(e)}")
                self.mineru_client = None
        else:
            self.logger.info("配置为使用PyPDF2提取")
    
    def run(self, pdf_files: List[str] = None) -> Dict[str, str]:
        """
        提取PDF文件内容
        
        Args:
            pdf_files: PDF文件路径列表，如果为None则自动读取data/input目录
            
        Returns:
            字典，格式为 {pdf_filename: extracted_text}
        """
        self.log_start("批量提取PDF文档")
        
        try:
            # 如果未提供文件列表，则从input目录读取
            if pdf_files is None:
                input_dir = self.file_manager.get_dir('input')
                pdf_files = [str(f) for f in input_dir.glob('*.pdf')]
                self.logger.info(f"从input目录找到 {len(pdf_files)} 个PDF文件")
            
            if not pdf_files:
                self.logger.warning("未找到任何PDF文件")
                return {}
            
            results = {}
            
            # 逐个处理PDF文件
            for pdf_path in pdf_files:
                self.logger.info(f"正在处理: {pdf_path}")
                
                try:
                    # 尝试使用MinerU API
                    if self.mineru_client:
                        extracted_text = self._extract_with_mineru(pdf_path)
                    else:
                        # 使用PyPDF2作为备选
                        extracted_text = self._extract_with_pypdf2(pdf_path)
                    
                    # 保存提取结果
                    pdf_name = Path(pdf_path).stem
                    output_filename = f"{pdf_name}_extracted.txt"
                    self.save_result(
                        extracted_text,
                        output_filename,
                        'extracted',
                        format='text'
                    )
                    
                    results[pdf_name] = extracted_text
                    self.logger.info(f"✓ {pdf_name} 提取完成")
                    
                except Exception as e:
                    self.log_error(f"提取 {pdf_path} 失败: {str(e)}")
                    continue
            
            self.log_end(f"批量提取PDF文档，成功: {len(results)}/{len(pdf_files)}")
            return results
            
        except Exception as e:
            self.log_error(f"批量提取失败: {str(e)}")
            raise
    
    def _extract_with_mineru(self, pdf_path: str) -> str:
        """
        使用MinerU API提取PDF
        
        Args:
            pdf_path: PDF文件路径
            
        Returns:
            提取的文本
        """
        pdf_path_obj = Path(pdf_path)
        pdf_name = pdf_path_obj.stem
        
        # 创建保存目录
        save_dir = self.file_manager.get_dir('extracted') / f"{pdf_name}_mineru"
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 检查是否已有URL（如果PDF文件是从URL下载的）
        # 这里我们需要先上传文件或使用已有的URL
        # 由于MinerU不支持直接上传，我们提供两种方式：
        
        # 方式1: 如果文件已经有公开URL（推荐）
        # 用户需要在调用时提供URL
        
        # 方式2: 使用批量上传接口（需要额外实现）
        
        # 这里我们抛出提示，让用户使用URL方式
        raise NotImplementedError(
            f"MinerU需要PDF文件的URL而非本地路径。\n"
            f"请使用以下方式之一：\n"
            f"1. 使用 extract_from_url() 方法并提供PDF的URL\n"
            f"2. 先将PDF上传到云存储获取URL\n"
            f"3. 使用 PyPDF2 作为备选方案（设置 use_mineru=False）"
        )
    
    def extract_from_url(
        self,
        pdf_url: str,
        pdf_name: Optional[str] = None,
        model_version: str = "vlm",
        enable_formula: bool = True,
        enable_table: bool = True
    ) -> str:
        """
        从URL提取PDF（使用MinerU）
        
        Args:
            pdf_url: PDF文件的URL
            pdf_name: PDF名称（用于保存），如果为None则从URL提取
            model_version: MinerU模型版本 ("pipeline" 或 "vlm")
            enable_formula: 是否识别公式
            enable_table: 是否识别表格
            
        Returns:
            提取的文本内容
        """
        if not self.mineru_client:
            raise ValueError("MinerU客户端未初始化，请检查API密钥")
        
        self.log_start(f"使用MinerU提取PDF: {pdf_url}")
        
        # 确定文件名
        if pdf_name is None:
            pdf_name = Path(pdf_url).stem or "unnamed_pdf"
        
        # 创建保存目录
        save_dir = self.file_manager.get_dir('extracted') / f"{pdf_name}_mineru"
        
        try:
            # 使用MinerU提取
            content = self.mineru_client.extract_pdf_from_url(
                file_url=pdf_url,
                save_dir=str(save_dir),
                model_version=model_version,
                enable_formula=enable_formula,
                enable_table=enable_table,
                data_id=pdf_name
            )
            
            # 保存提取的文本
            output_filename = f"{pdf_name}_extracted.txt"
            self.save_result(content, output_filename, 'extracted', format='text')
            
            self.log_end(f"提取完成: {pdf_name}")
            return content
            
        except Exception as e:
            self.log_error(f"MinerU提取失败: {str(e)}")
            raise
    
    def extract_from_urls(
        self,
        pdf_urls: List[str],
        pdf_names: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, str]:
        """
        批量从URL提取PDF（使用MinerU批量接口）
        
        Args:
            pdf_urls: PDF URL列表
            pdf_names: PDF名称列表（可选）
            **kwargs: 其他参数
            
        Returns:
            提取结果字典 {pdf_name: content}
        """
        if not self.mineru_client:
            raise ValueError("MinerU客户端未初始化")
        
        self.log_start(f"批量提取 {len(pdf_urls)} 个PDF")
        
        # 准备文件列表
        files = []
        if pdf_names is None:
            pdf_names = [Path(url).stem or f"pdf_{i}" for i, url in enumerate(pdf_urls)]
        
        for url, name in zip(pdf_urls, pdf_names):
            files.append({
                "url": url,
                "data_id": name
            })
        
        try:
            # 创建批量任务
            batch_id = self.mineru_client.batch_create_tasks(files, **kwargs)
            
            # 等待完成
            results = self.mineru_client.wait_for_batch(batch_id)
            
            # 处理结果
            extracted_contents = {}
            
            for result in results:
                file_name = result.get("file_name", "")
                state = result.get("state")
                data_id = result.get("data_id", Path(file_name).stem)
                
                if state == "done":
                    zip_url = result.get("full_zip_url")
                    if zip_url:
                        # 下载并提取内容
                        save_dir = self.file_manager.get_dir('extracted') / f"{data_id}_mineru"
                        files = self.mineru_client.download_result(zip_url, str(save_dir))
                        
                        if files.get('markdown'):
                            with open(files['markdown'], 'r', encoding='utf-8') as f:
                                content = f.read()
                            
                            # 保存
                            output_filename = f"{data_id}_extracted.txt"
                            self.save_result(content, output_filename, 'extracted', format='text')
                            
                            extracted_contents[data_id] = content
                            self.logger.info(f"  ✓ {data_id} 提取成功")
                
                elif state == "failed":
                    err_msg = result.get("err_msg", "未知错误")
                    self.logger.error(f"  ✗ {data_id} 提取失败: {err_msg}")
            
            self.log_end(f"批量提取完成，成功: {len(extracted_contents)}/{len(pdf_urls)}")
            return extracted_contents
            
        except Exception as e:
            self.log_error(f"批量提取失败: {str(e)}")
            raise
    
    def _extract_with_pypdf2(self, pdf_path: str) -> str:
        """
        使用PyPDF2提取PDF（备选方案）
        
        Args:
            pdf_path: PDF文件路径
            
        Returns:
            提取的文本
        """
        self.logger.info(f"使用PyPDF2提取: {pdf_path}")
        
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text_parts = []
            
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text_parts.append(page.extract_text())
            
            return '\n\n'.join(text_parts)
    
    def extract_single(self, pdf_path: str) -> str:
        """
        提取单个PDF文件
        
        Args:
            pdf_path: PDF文件路径
            
        Returns:
            提取的文本内容
        """
        self.log_start(f"提取单个PDF: {pdf_path}")
        
        result = self.run([pdf_path])
        pdf_name = Path(pdf_path).stem
        
        return result.get(pdf_name, "")

