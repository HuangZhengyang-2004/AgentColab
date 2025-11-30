"""
MinerU API客户端
完整实现MinerU PDF解析功能
"""

import os
import time
import requests
import zipfile
import json
from pathlib import Path
from typing import Dict, List, Optional, Union
from dataclasses import dataclass

from utils.logger import logger


@dataclass
class TaskStatus:
    """任务状态数据类"""
    task_id: str
    state: str  # pending, running, converting, done, failed
    full_zip_url: Optional[str] = None
    err_msg: Optional[str] = None
    data_id: Optional[str] = None
    extract_progress: Optional[Dict] = None


class MinerUClient:
    """MinerU API客户端"""
    
    # API端点
    BASE_URL = "https://mineru.net/api/v4"
    TASK_CREATE_URL = f"{BASE_URL}/extract/task"
    TASK_QUERY_URL = f"{BASE_URL}/extract/task"
    BATCH_FILE_URL = f"{BASE_URL}/file-urls/batch"
    BATCH_TASK_URL = f"{BASE_URL}/extract/task/batch"
    BATCH_RESULT_URL = f"{BASE_URL}/extract-results/batch"
    
    def __init__(self, api_key: str, timeout: int = 300):
        """
        初始化MinerU客户端
        
        Args:
            api_key: MinerU API密钥
            timeout: 请求超时时间（秒）
        """
        if not api_key:
            raise ValueError("MinerU API密钥未设置")
        
        self.api_key = api_key
        self.timeout = timeout
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "Accept": "*/*"
        }
        
        logger.info("MinerU客户端初始化成功")
    
    def create_task(
        self,
        file_url: str,
        model_version: str = "vlm",
        data_id: Optional[str] = None,
        is_ocr: bool = False,
        enable_formula: bool = True,
        enable_table: bool = True,
        language: str = "ch",
        page_ranges: Optional[str] = None,
        extra_formats: Optional[List[str]] = None
    ) -> Dict:
        """
        创建单个文件解析任务
        
        Args:
            file_url: PDF文件的URL
            model_version: 模型版本 ("pipeline" 或 "vlm")
            data_id: 数据ID，用于标识业务数据
            is_ocr: 是否启用OCR
            enable_formula: 是否开启公式识别
            enable_table: 是否开启表格识别
            language: 文档语言
            page_ranges: 页码范围，如 "1-10,15,20-30"
            extra_formats: 额外导出格式，如 ["docx", "html"]
            
        Returns:
            包含task_id的响应字典
        """
        logger.info(f"创建解析任务: {file_url}")
        
        payload = {
            "url": file_url,
            "model_version": model_version,
        }
        
        # 添加可选参数
        if data_id:
            payload["data_id"] = data_id
        if is_ocr:
            payload["is_ocr"] = is_ocr
        if enable_formula:
            payload["enable_formula"] = enable_formula
        if enable_table:
            payload["enable_table"] = enable_table
        if language:
            payload["language"] = language
        if page_ranges:
            payload["page_ranges"] = page_ranges
        if extra_formats:
            payload["extra_formats"] = extra_formats
        
        try:
            response = requests.post(
                self.TASK_CREATE_URL,
                headers=self.headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get("code") == 0:
                    task_id = result["data"]["task_id"]
                    logger.info(f"✓ 任务创建成功，task_id: {task_id}")
                    return result
                else:
                    error_msg = result.get("msg", "未知错误")
                    logger.error(f"创建任务失败: {error_msg}")
                    raise Exception(f"MinerU API错误: {error_msg}")
            else:
                logger.error(f"HTTP错误: {response.status_code}")
                raise Exception(f"HTTP {response.status_code}: {response.text}")
                
        except requests.exceptions.RequestException as e:
            logger.error(f"请求失败: {str(e)}")
            raise
    
    def get_task_status(self, task_id: str) -> TaskStatus:
        """
        查询任务状态
        
        Args:
            task_id: 任务ID
            
        Returns:
            TaskStatus对象
        """
        url = f"{self.TASK_QUERY_URL}/{task_id}"
        
        try:
            response = requests.get(
                url,
                headers=self.headers,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get("code") == 0:
                    data = result["data"]
                    return TaskStatus(
                        task_id=data.get("task_id"),
                        state=data.get("state"),
                        full_zip_url=data.get("full_zip_url"),
                        err_msg=data.get("err_msg"),
                        data_id=data.get("data_id"),
                        extract_progress=data.get("extract_progress")
                    )
                else:
                    error_msg = result.get("msg", "未知错误")
                    raise Exception(f"查询任务失败: {error_msg}")
            else:
                raise Exception(f"HTTP {response.status_code}: {response.text}")
                
        except requests.exceptions.RequestException as e:
            logger.error(f"查询任务状态失败: {str(e)}")
            raise
    
    def wait_for_task(
        self,
        task_id: str,
        max_wait_time: int = 600,
        poll_interval: int = 5
    ) -> TaskStatus:
        """
        等待任务完成
        
        Args:
            task_id: 任务ID
            max_wait_time: 最大等待时间（秒）
            poll_interval: 轮询间隔（秒）
            
        Returns:
            TaskStatus对象
        """
        logger.info(f"等待任务完成: {task_id}")
        
        start_time = time.time()
        last_progress = None
        
        while time.time() - start_time < max_wait_time:
            status = self.get_task_status(task_id)
            
            # 显示进度
            if status.state == "running" and status.extract_progress:
                progress = status.extract_progress
                current = progress.get("extracted_pages", 0)
                total = progress.get("total_pages", 0)
                
                if progress != last_progress:
                    logger.info(f"  解析进度: {current}/{total} 页")
                    last_progress = progress
            
            elif status.state == "pending":
                logger.info("  任务排队中...")
            
            elif status.state == "converting":
                logger.info("  格式转换中...")
            
            elif status.state == "done":
                logger.info(f"✓ 任务完成！")
                return status
            
            elif status.state == "failed":
                error_msg = status.err_msg or "未知错误"
                logger.error(f"✗ 任务失败: {error_msg}")
                raise Exception(f"任务失败: {error_msg}")
            
            time.sleep(poll_interval)
        
        raise TimeoutError(f"任务超时（{max_wait_time}秒）")
    
    def download_result(
        self,
        zip_url: str,
        save_dir: str,
        extract: bool = True
    ) -> Dict[str, str]:
        """
        下载并解压结果
        
        Args:
            zip_url: ZIP文件URL
            save_dir: 保存目录
            extract: 是否解压
            
        Returns:
            包含文件路径的字典
        """
        logger.info(f"下载解析结果: {zip_url}")
        
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # 下载ZIP文件
        zip_file = save_path / "result.zip"
        
        try:
            response = requests.get(zip_url, timeout=self.timeout, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(zip_file, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            progress = (downloaded / total_size) * 100
                            if downloaded % (1024 * 1024) == 0:  # 每1MB显示一次
                                logger.info(f"  下载进度: {progress:.1f}%")
            
            logger.info(f"✓ 下载完成: {zip_file}")
            
            # 解压
            if extract:
                extract_dir = save_path / "extracted"
                extract_dir.mkdir(exist_ok=True)
                
                with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                    zip_ref.extractall(extract_dir)
                
                logger.info(f"✓ 解压完成: {extract_dir}")
                
                # 查找并返回主要文件
                files = {
                    'markdown': None,
                    'json': None,
                    'images_dir': None
                }
                
                for file_path in extract_dir.rglob('*'):
                    if file_path.suffix == '.md':
                        files['markdown'] = str(file_path)
                    elif file_path.suffix == '.json':
                        files['json'] = str(file_path)
                    elif file_path.is_dir() and 'images' in file_path.name:
                        files['images_dir'] = str(file_path)
                
                return files
            
            return {'zip_file': str(zip_file)}
            
        except Exception as e:
            logger.error(f"下载结果失败: {str(e)}")
            raise
    
    def extract_pdf_from_url(
        self,
        file_url: str,
        save_dir: str,
        **kwargs
    ) -> str:
        """
        从URL提取PDF内容（完整流程）
        
        Args:
            file_url: PDF文件URL
            save_dir: 保存目录
            **kwargs: 其他参数传递给create_task
            
        Returns:
            提取的Markdown文本
        """
        # 1. 创建任务
        result = self.create_task(file_url, **kwargs)
        task_id = result["data"]["task_id"]
        
        # 2. 等待完成
        status = self.wait_for_task(task_id)
        
        # 3. 下载结果
        files = self.download_result(status.full_zip_url, save_dir)
        
        # 4. 读取Markdown内容
        if files.get('markdown'):
            with open(files['markdown'], 'r', encoding='utf-8') as f:
                content = f.read()
            logger.info(f"✓ 提取完成，内容长度: {len(content)} 字符")
            return content
        else:
            raise Exception("未找到Markdown文件")
    
    def upload_and_extract_file(
        self,
        file_path: str,
        data_id: Optional[str] = None,
        model_version: str = "vlm",
        **kwargs
    ) -> str:
        """
        上传本地文件并提取内容（完整流程）
        
        Args:
            file_path: 本地PDF文件路径
            data_id: 数据ID
            model_version: 模型版本
            **kwargs: 其他参数
            
        Returns:
            提取的Markdown文本
        """
        logger.info(f"上传并提取本地文件: {file_path}")
        
        # 1. 申请上传链接
        file_name = Path(file_path).name
        if not data_id:
            data_id = Path(file_path).stem
        
        files_param = [{
            "name": file_name,
            "data_id": data_id
        }]
        
        payload = {
            "files": files_param,
            "model_version": model_version,
            **kwargs
        }
        
        try:
            # 申请上传URL
            logger.info("申请上传链接...")
            response = requests.post(
                self.BATCH_FILE_URL,
                headers=self.headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get("code") == 0:
                    batch_id = result["data"]["batch_id"]
                    upload_urls = result["data"]["file_urls"]
                    logger.info(f"✓ 获取上传链接成功，batch_id: {batch_id}")
                else:
                    raise Exception(f"申请上传链接失败: {result.get('msg')}")
            else:
                raise Exception(f"HTTP {response.status_code}: {response.text}")
            
            # 2. 上传文件
            logger.info("上传文件到MinerU服务器...")
            upload_url = upload_urls[0]
            
            with open(file_path, 'rb') as f:
                upload_response = requests.put(
                    upload_url,
                    data=f,
                    timeout=self.timeout
                )
                
                if upload_response.status_code == 200:
                    logger.info("✓ 文件上传成功")
                else:
                    raise Exception(f"文件上传失败: HTTP {upload_response.status_code}")
            
            # 3. 等待解析完成
            logger.info("等待MinerU解析...")
            results = self.wait_for_batch(batch_id)
            
            # 4. 下载结果
            for result in results:
                if result.get("state") == "done" and result.get("data_id") == data_id:
                    zip_url = result.get("full_zip_url")
                    if zip_url:
                        save_dir = Path("data/extracted") / f"{data_id}_mineru"
                        files = self.download_result(zip_url, str(save_dir))
                        
                        if files.get('markdown'):
                            with open(files['markdown'], 'r', encoding='utf-8') as f:
                                content = f.read()
                            logger.info(f"✓ 提取完成，内容长度: {len(content)} 字符")
                            return content
            
            raise Exception("未找到解析结果")
            
        except Exception as e:
            logger.error(f"上传并提取失败: {str(e)}")
            raise
    
    def batch_create_tasks(
        self,
        file_urls: List[Dict[str, str]],
        model_version: str = "vlm",
        **kwargs
    ) -> str:
        """
        批量创建解析任务（URL方式）
        
        Args:
            file_urls: 文件URL列表，格式: [{"url": "...", "data_id": "..."}]
            model_version: 模型版本
            **kwargs: 其他参数
            
        Returns:
            batch_id
        """
        logger.info(f"批量创建解析任务，数量: {len(file_urls)}")
        
        payload = {
            "files": file_urls,
            "model_version": model_version,
            **kwargs
        }
        
        try:
            response = requests.post(
                self.BATCH_TASK_URL,
                headers=self.headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get("code") == 0:
                    batch_id = result["data"]["batch_id"]
                    logger.info(f"✓ 批量任务创建成功，batch_id: {batch_id}")
                    return batch_id
                else:
                    raise Exception(f"创建失败: {result.get('msg')}")
            else:
                raise Exception(f"HTTP {response.status_code}: {response.text}")
                
        except requests.exceptions.RequestException as e:
            logger.error(f"批量创建任务失败: {str(e)}")
            raise
    
    def get_batch_results(self, batch_id: str) -> List[Dict]:
        """
        获取批量任务结果
        
        Args:
            batch_id: 批次ID
            
        Returns:
            结果列表
        """
        url = f"{self.BATCH_RESULT_URL}/{batch_id}"
        
        try:
            response = requests.get(
                url,
                headers=self.headers,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get("code") == 0:
                    return result["data"]["extract_result"]
                else:
                    raise Exception(f"查询失败: {result.get('msg')}")
            else:
                raise Exception(f"HTTP {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            logger.error(f"获取批量结果失败: {str(e)}")
            raise
    
    def wait_for_batch(
        self,
        batch_id: str,
        max_wait_time: int = 1800,
        poll_interval: int = 10
    ) -> List[Dict]:
        """
        等待批量任务完成
        
        Args:
            batch_id: 批次ID
            max_wait_time: 最大等待时间
            poll_interval: 轮询间隔
            
        Returns:
            完成的任务列表
        """
        logger.info(f"等待批量任务完成: {batch_id}")
        
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            results = self.get_batch_results(batch_id)
            
            # 统计状态
            status_count = {}
            for item in results:
                state = item.get("state", "unknown")
                status_count[state] = status_count.get(state, 0) + 1
            
            logger.info(f"  状态统计: {status_count}")
            
            # 检查是否全部完成
            all_done = all(
                item.get("state") in ["done", "failed"]
                for item in results
            )
            
            if all_done:
                done_count = status_count.get("done", 0)
                failed_count = status_count.get("failed", 0)
                logger.info(f"✓ 批量任务完成！成功: {done_count}, 失败: {failed_count}")
                return results
            
            time.sleep(poll_interval)
        
        raise TimeoutError(f"批量任务超时（{max_wait_time}秒）")


def get_mineru_client(api_key: Optional[str] = None) -> MinerUClient:
    """
    获取MinerU客户端实例
    
    Args:
        api_key: API密钥，如果为None则从环境变量读取
        
    Returns:
        MinerUClient实例
    """
    if api_key is None:
        api_key = os.getenv('MINERU_API_KEY', '')
    
    return MinerUClient(api_key)

