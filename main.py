# main.py (主程序)
import time  # 导入时间模块，用于延时和时间操作
import schedule  # 导入调度模块，用于定时任务
from PIL import Image  # 导入PIL库的Image模块，用于图像处理
import pytesseract  # 导入pytesseract，用于OCR文字识别
import numpy as np  # 导入numpy，用于数值计算和数组操作
import cv2  # 导入opencv，用于计算机视觉和图像处理
from datetime import datetime, timedelta  # 导入日期时间模块
import mss  # 导入mss，用于快速屏幕截图
import logging  # 导入日志模块，用于记录程序运行信息
import os  # 导入操作系统接口模块
from pathlib import Path  # 导入路径处理模块
import pyautogui  # 导入pyautogui，用于自动化鼠标键盘操作
from typing import List, Dict, Callable, Optional, Any  # 导入类型提示模块
import importlib.util  # 导入模块导入工具，用于动态导入配置文件

# 从外部文件导入配置
def load_config():
    """从config.py文件加载配置"""
    config_path = Path("config.py")  # 创建配置文件路径对象
    if not config_path.exists():  # 检查配置文件是否存在
        raise FileNotFoundError(f"配置文件 {config_path} 不存在")  # 如果不存在则抛出异常
    
    spec = importlib.util.spec_from_file_location("config", config_path)  # 创建模块规格
    config = importlib.util.module_from_spec(spec)  # 创建模块对象
    spec.loader.exec_module(config)  # 执行模块代码
    
    return config.GLOBAL_CONFIG, config.RECOGNITION_CONFIGS  # 返回全局配置和识别配置

# 加载配置
try:
    GLOBAL_CONFIG, RECOGNITION_CONFIGS = load_config()  # 尝试加载配置文件
except Exception as e:  # 捕获加载配置时可能发生的异常
    print(f"加载配置文件失败: {e}")  # 打印错误信息
    print("请确保config.py文件存在且格式正确")  # 提示用户检查配置文件
    exit(1)  # 退出程序

# 配置日志
logging.basicConfig(
    level=logging.INFO,  # 设置日志级别为INFO
    format='%(asctime)s - %(levelname)s - %(message)s',  # 设置日志格式
    handlers=[  # 设置日志处理器
        logging.FileHandler('screenshot_ocr.log'),  # 文件处理器，将日志写入文件
        logging.StreamHandler()  # 控制台处理器，将日志输出到控制台
    ]
)

class RecognitionConfig:
    """识别配置项"""
    def __init__(self, config_data: Dict):
        """初始化识别配置"""
        self.enabled = config_data.get("enabled", True)  # 获取启用状态，默认为True
        self.name = config_data["name"]  # 获取配置名称
        self.type = config_data.get("type", "text")  # 获取识别类型，默认为"text"
        self.x = config_data["x"]  # 获取区域X坐标
        self.y = config_data["y"]  # 获取区域Y坐标
        self.width = config_data["width"]  # 获取区域宽度
        self.height = config_data["height"]  # 获取区域高度
        self.callback_type = config_data["callback_type"]  # 获取回调类型
        self.callback_params = config_data["callback_params"]  # 获取回调参数
        
        # 文字识别相关配置
        if self.type == "text":  # 如果是文字识别类型
            self.keywords = config_data.get("keywords", [])  # 获取关键词列表，默认为空列表
            self.case_sensitive = config_data.get("case_sensitive", False)  # 获取是否区分大小写，默认为False
            self.ocr_lang = config_data.get("ocr_lang", GLOBAL_CONFIG["ocr_lang"])  # 获取OCR语言，默认使用全局配置
        
        # 模板匹配相关配置
        elif self.type == "template":  # 如果是模板匹配类型
            self.template_path = config_data["template_path"]  # 获取模板图片路径
            self.match_threshold = config_data.get("match_threshold", 0.8)  # 获取匹配阈值，默认为0.8
            self.template_image = self._load_template()  # 加载模板图片
    
    def _load_template(self) -> Optional[np.ndarray]:
        """加载模板图片"""
        try:
            template_path = Path(self.template_path)  # 创建模板图片路径对象
            if not template_path.exists():  # 检查模板图片文件是否存在
                logging.error(f"模板图片不存在: {self.template_path}")  # 记录错误日志
                return None  # 返回None
            
            template = cv2.imread(str(template_path), cv2.IMREAD_GRAYSCALE)  # 读取模板图片为灰度图
            if template is None:  # 检查图片是否成功读取
                logging.error(f"无法读取模板图片: {self.template_path}")  # 记录错误日志
                return None  # 返回None
            
            logging.info(f"模板图片已加载: {self.template_path}")  # 记录信息日志
            return template  # 返回模板图片数据
            
        except Exception as e:  # 捕获加载过程中可能发生的异常
            logging.error(f"加载模板图片失败 {self.template_path}: {e}")  # 记录错误日志
            return None  # 返回None

class DesktopMonitor:
    def __init__(self):
        """
        初始化桌面监控器
        """
        self.screenshot_dir = Path(GLOBAL_CONFIG["screenshot_dir"])  # 创建截图目录路径对象
        self.screenshot_dir.mkdir(exist_ok=True)  # 创建截图目录，如果已存在则不报错
        
        self.sct = mss.mss()  # 创建mss截图对象
        
        # 验证Tesseract
        try:
            pytesseract.get_tesseract_version()  # 尝试获取Tesseract版本
            logging.info("Tesseract OCR 初始化成功")  # 记录信息日志
        except Exception as e:  # 捕获初始化失败的异常
            logging.error(f"Tesseract OCR 初始化失败: {e}")  # 记录错误日志
            raise  # 重新抛出异常
        
        # 创建配置实例
        self.recognition_configs = self._create_configs()  # 创建识别配置实例列表
        
        # 配置pyautogui
        pyautogui.FAILSAFE = True  # 启用安全模式，鼠标移到屏幕左上角可中断程序
        pyautogui.PAUSE = 0.5  # 设置pyautogui操作之间的默认暂停时间
    
    def _create_configs(self) -> List[RecognitionConfig]:
        """创建配置实例"""
        configs = []  # 创建空的配置列表
        for config_data in RECOGNITION_CONFIGS:  # 遍历配置数据
            if config_data.get("enabled", True):  # 检查配置是否启用
                config = RecognitionConfig(config_data)  # 创建配置实例
                configs.append(config)  # 添加到配置列表
                logging.info(f"已创建配置: {config.name}")  # 记录信息日志
            else:
                logging.info(f"跳过禁用的配置: {config_data['name']}")  # 记录跳过禁用配置的信息
        
        return configs  # 返回配置实例列表
    
    def create_mouse_click_callback(self, x: int, y: int, clicks: int = 1) -> Callable:
        """创建鼠标点击回调函数"""
        def callback(context: Dict):
            logging.info(f"执行鼠标点击回调: 点击({x}, {y})")  # 记录信息日志
            self.perform_mouse_click(x, y, clicks)  # 执行鼠标点击
        return callback  # 返回回调函数
    
    def create_keyboard_input_callback(self, text: str) -> Callable:
        """创建键盘输入回调函数"""
        def callback(context: Dict):
            logging.info(f"执行键盘输入回调: 输入'{text}'")  # 记录信息日志
            self.perform_keyboard_input(text)  # 执行键盘输入
        return callback  # 返回回调函数
    
    def create_logging_callback(self, message: str = None) -> Callable:
        """创建日志记录回调函数"""
        def callback(context: Dict):
            msg = message or f"检测到匹配: {context.get('match_type')}='{context.get('match_value')}'"  # 设置日志消息
            logging.info(msg)  # 记录信息日志
        return callback  # 返回回调函数
    
    def capture_screen_region(self, x: int, y: int, width: int, height: int) -> Optional[np.ndarray]:
        """截取屏幕指定区域"""
        try:
            monitor = {  # 创建监控区域字典
                "top": y,  # 顶部坐标
                "left": x,  # 左侧坐标
                "width": width,  # 宽度
                "height": height  # 高度
            }
            
            screenshot = self.sct.grab(monitor)  # 执行截图
            img = np.array(screenshot)  # 转换为numpy数组
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)  # 转换颜色格式
            
            return img  # 返回图像数据
            
        except Exception as e:  # 捕获截图过程中的异常
            logging.error(f"截图失败: {e}")  # 记录错误日志
            return None  # 返回None
    
    def preprocess_image_for_ocr(self, img: np.ndarray) -> np.ndarray:
        """为OCR识别预处理图像"""
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # 转换为灰度图
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))  # 创建CLAHE对象
            enhanced = clahe.apply(gray)  # 应用CLAHE增强
            _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # 二值化处理
            scale_factor = 2  # 设置缩放因子
            binary = cv2.resize(binary, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)  # 放大图像
            
            return binary  # 返回预处理后的图像
            
        except Exception as e:  # 捕获预处理过程中的异常
            logging.error(f"图像预处理失败: {e}")  # 记录错误日志
            return img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # 返回原始图像或灰度图
    
    def preprocess_image_for_template(self, img: np.ndarray) -> np.ndarray:
        """为模板匹配预处理图像"""
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # 转换为灰度图
            # 可以添加更多模板匹配专用的预处理
            return gray  # 返回灰度图
        except Exception as e:  # 捕获预处理过程中的异常
            logging.error(f"模板匹配图像预处理失败: {e}")  # 记录错误日志
            return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # 返回灰度图
    
    def extract_text(self, img: np.ndarray, ocr_lang: str) -> str:
        """从图像中提取文字"""
        try:
            custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ --dpi 300'  # OCR配置参数
            
            text = pytesseract.image_to_string(img, lang=ocr_lang, config=custom_config)  # 执行OCR识别
            return text.strip()  # 返回去除首尾空白的文字
            
        except Exception as e:  # 捕获OCR识别过程中的异常
            logging.error(f"OCR识别失败: {e}")  # 记录错误日志
            return ""  # 返回空字符串
    
    def match_template(self, img: np.ndarray, template: np.ndarray, threshold: float) -> bool:
        """模板匹配"""
        try:
            if template is None:  # 检查模板是否为空
                return False  # 返回False
            
            # 执行模板匹配
            result = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)  # 执行模板匹配
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)  # 获取匹配结果
            
            # 检查匹配度是否超过阈值
            if max_val >= threshold:  # 如果最大匹配值大于等于阈值
                logging.info(f"模板匹配成功: 匹配度={max_val:.3f}, 阈值={threshold}")  # 记录成功日志
                return True  # 返回True
            else:  # 匹配度不足
                logging.info(f"模板匹配失败: 匹配度={max_val:.3f}, 阈值={threshold}")  # 记录失败日志
                return False  # 返回False
                
        except Exception as e:  # 捕获模板匹配过程中的异常
            logging.error(f"模板匹配失败: {e}")  # 记录错误日志
            return False  # 返回False
    
    def save_screenshot(self, img: np.ndarray, timestamp: str, context: Dict[str, Any]) -> Optional[Path]:
        """保存截图"""
        try:
            config_name = context.get('config_name', 'unknown')  # 获取配置名称
            match_type = context.get('match_type', 'unknown')  # 获取匹配类型
            match_value = context.get('match_value', 'unknown')  # 获取匹配值
            safe_value = "".join(c for c in str(match_value) if c.isalnum() or c in ' _-')[:50]  # 创建安全的文件名
            
            filename = f"{config_name}_{match_type}_{timestamp}_{safe_value}.png"  # 构建文件名
            filepath = self.screenshot_dir / filename  # 创建文件路径
            
            debug_img = Image.fromarray(img)  # 将numpy数组转换为PIL图像
            debug_img.save(filepath, quality=95)  # 保存图像
            
            logging.info(f"截图已保存: {filepath}")  # 记录信息日志
            return filepath  # 返回文件路径
            
        except Exception as e:  # 捕获保存过程中的异常
            logging.error(f"保存截图失败: {e}")  # 记录错误日志
            return None  # 返回None
    
    def perform_mouse_click(self, x: int, y: int, clicks: int = 1, interval: float = 0.1):
        """执行鼠标点击"""
        try:
            pyautogui.click(x, y, clicks=clicks, interval=interval)  # 执行鼠标点击
            logging.info(f"鼠标已点击坐标: ({x}, {y})")  # 记录信息日志
            return True  # 返回True表示成功
        except Exception as e:  # 捕获鼠标点击过程中的异常
            logging.error(f"鼠标点击失败: {e}")  # 记录错误日志
            return False  # 返回False表示失败
    
    def perform_keyboard_input(self, text: str):
        """执行键盘输入"""
        try:
            pyautogui.typewrite(text)  # 执行键盘输入
            logging.info(f"已输入文本: {text}")  # 记录信息日志
            return True  # 返回True表示成功
        except Exception as e:  # 捕获键盘输入过程中的异常
            logging.error(f"键盘输入失败: {e}")  # 记录错误日志
            return False  # 返回False表示失败
    
    def clean_old_screenshots(self, retention_hours: int = 24):
        """清理过期截图"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=retention_hours)  # 计算截止时间
            deleted_count = 0  # 删除计数器
            
            for file_path in self.screenshot_dir.glob("*.png"):  # 遍历所有png文件
                try:
                    file_mtime = datetime.fromtimestamp(file_path.stat().st_mtime)  # 获取文件修改时间
                    if file_mtime < cutoff_time:  # 如果文件修改时间早于截止时间
                        file_path.unlink()  # 删除文件
                        logging.info(f"已删除过期截图: {file_path}")  # 记录信息日志
                        deleted_count += 1  # 增加删除计数
                except (ValueError, OSError) as e:  # 捕获文件处理过程中的异常
                    logging.warning(f"处理文件 {file_path} 时出错: {e}")  # 记录警告日志
                    continue  # 继续处理下一个文件
            
            if deleted_count > 0:  # 如果有文件被删除
                logging.info(f"清理完成，共删除 {deleted_count} 个过期文件")  # 记录信息日志
                
        except Exception as e:  # 捕获清理过程中的异常
            logging.error(f"清理旧截图时出错: {e}")  # 记录错误日志
    
    def create_callback(self, config: RecognitionConfig) -> Callable:
        """根据配置创建回调函数"""
        if config.callback_type == "mouse_click":  # 如果回调类型为鼠标点击
            return self.create_mouse_click_callback(
                config.callback_params["x"],  # X坐标
                config.callback_params["y"]  # Y坐标
            )
        elif config.callback_type == "keyboard_input":  # 如果回调类型为键盘输入
            return self.create_keyboard_input_callback(
                config.callback_params["text"]  # 要输入的文本
            )
        elif config.callback_type == "logging":  # 如果回调类型为日志记录
            return self.create_logging_callback(
                config.callback_params.get("message")  # 日志消息
            )
        else:  # 其他未知回调类型
            # 默认日志回调
            return self.create_logging_callback(
                f"未知回调类型: {config.callback_type}"  # 提示未知类型
            )
    
    def monitor_once(self):
        """执行一次监控任务"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # 生成时间戳
            logging.info(f"开始执行监控任务: {timestamp}")  # 记录信息日志
            
            # 对每个启用的配置进行处理
            for config in self.recognition_configs:  # 遍历所有配置
                logging.info(f"处理识别区域: {config.name}")  # 记录正在处理的配置
                
                # 1. 截图
                img = self.capture_screen_region(config.x, config.y, config.width, config.height)  # 截取指定区域
                if img is None:  # 如果截图失败
                    continue  # 跳过当前配置
                
                # 2. 根据类型进行预处理
                if config.type == "text":  # 如果是文字识别类型
                    processed_img = self.preprocess_image_for_ocr(img)  # 使用OCR预处理
                else:  # template 模板匹配类型
                    processed_img = self.preprocess_image_for_template(img)  # 使用模板匹配预处理
                
                # 3. 匹配检测
                match_found = False  # 匹配标志
                match_value = None  # 匹配值
                
                if config.type == "text":  # 文字识别
                    # 文字识别
                    text = self.extract_text(processed_img, config.ocr_lang)  # 执行OCR识别
                    if text:  # 如果识别到文字
                        logging.info(f"[{config.name}] 识别到的文字: '{text}'")  # 记录识别结果
                        
                        # 检查关键词匹配
                        search_text = text if config.case_sensitive else text.lower()  # 根据是否区分大小写处理文本
                        for keyword in config.keywords:  # 遍历关键词列表
                            search_keyword = keyword if config.case_sensitive else keyword.lower()  # 处理关键词大小写
                            if search_keyword in search_text:  # 检查是否包含关键词
                                match_found = True  # 设置匹配标志
                                match_value = text  # 保存匹配值
                                logging.info(f"[{config.name}] 检测到关键词 '{keyword}'")  # 记录匹配信息
                                break  # 找到匹配就退出循环
                    else:  # 未识别到文字
                        logging.info(f"[{config.name}] 未识别到有效文字")  # 记录信息日志
                
                elif config.type == "template":  # 模板匹配
                    # 模板匹配
                    if config.template_image is not None:  # 检查模板图片是否加载成功
                        match_found = self.match_template(
                            processed_img,  # 当前截图
                            config.template_image,  # 模板图片
                            config.match_threshold  # 匹配阈值
                        )
                        if match_found:  # 如果匹配成功
                            match_value = f"匹配度_{config.match_threshold}"  # 设置匹配值
                
                # 4. 如果匹配成功，执行回调
                if match_found:  # 如果找到匹配
                    context = {  # 创建上下文信息
                        'config_name': config.name,  # 配置名称
                        'timestamp': timestamp,  # 时间戳
                        'match_type': config.type,  # 匹配类型
                        'match_value': match_value,  # 匹配值
                        'x': config.x,  # X坐标
                        'y': config.y,  # Y坐标
                        'width': config.width,  # 宽度
                        'height': config.height  # 高度
                    }
                    
                    # 创建并执行回调函数
                    callback = self.create_callback(config)  # 创建回调函数
                    try:
                        callback(context)  # 执行回调
                    except Exception as e:  # 捕获回调执行过程中的异常
                        logging.error(f"执行回调函数失败: {e}")  # 记录错误日志
                
                # 5. 保存截图
                screenshot_path = self.save_screenshot(processed_img, timestamp, {
                    'config_name': config.name,  # 配置名称
                    'timestamp': timestamp,  # 时间戳
                    'match_type': config.type,  # 匹配类型
                    'match_value': match_value if match_found else 'no_match'  # 匹配值或"no_match"
                })
                if screenshot_path:  # 如果截图保存成功
                    logging.info(f"[{config.name}] 截图已保存至: {screenshot_path}")  # 记录信息日志
                    
        except Exception as e:  # 捕获监控过程中的异常
            logging.error(f"监控任务执行失败: {e}")  # 记录错误日志
    
    def start_cleanup_scheduler(self):
        """启动自动清理调度器"""
        schedule.every(GLOBAL_CONFIG["cleanup_interval_hours"]).hours.do(
            self.clean_old_screenshots,  # 要执行的清理函数
            retention_hours=GLOBAL_CONFIG["retention_hours"]  # 保留时间参数
        )
        logging.info(f"已设置自动清理任务，每 {GLOBAL_CONFIG['cleanup_interval_hours']} 小时检查一次过期文件")  # 记录信息日志
    
    def start_monitoring(self):
        """开始持续监控"""
        if not self.recognition_configs:  # 如果没有启用的配置
            logging.warning("警告: 没有启用任何识别配置")  # 记录警告日志
            return  # 退出函数
        
        logging.info(f"启动桌面监控服务，每{GLOBAL_CONFIG['monitor_interval_minutes']}分钟执行一次")  # 记录信息日志
        logging.info(f"截图保存目录: {self.screenshot_dir.absolute()}")  # 记录截图目录
        logging.info(f"截图保留时间: {GLOBAL_CONFIG['retention_hours']}小时")  # 记录保留时间
        logging.info("安全模式: 启用（将鼠标移到屏幕左上角可中断程序）")  # 记录安全模式信息
        logging.info(f"启用的识别配置数量: {len(self.recognition_configs)}")  # 记录启用的配置数量
        
        for config in self.recognition_configs:  # 遍历启用的配置
            if config.type == "text":  # 文字识别配置
                keywords_str = ", ".join(config.keywords)  # 将关键词列表转换为字符串
                logging.info(f"  - {config.name}: ({config.x},{config.y},{config.width},{config.height}) "
                            f"类型: 文字识别, 关键词: [{keywords_str}]")  # 记录配置信息
            else:  # 模板匹配配置
                logging.info(f"  - {config.name}: ({config.x},{config.y},{config.width},{config.height}) "
                            f"类型: 模板匹配, 模板: {config.template_path}, 阈值: {config.match_threshold}")  # 记录配置信息
        
        logging.info("按 Ctrl+C 停止监控")  # 提示用户如何停止监控
        
        # 启动自动清理任务
        self.start_cleanup_scheduler()  # 启动清理调度器
        
        # 立即执行第一次监控
        self.monitor_once()  # 执行一次监控
        
        # 主监控循环
        try:
            while True:  # 无限循环
                schedule.run_pending()  # 执行待处理的调度任务
                time.sleep(1)  # 暂停1秒
        except KeyboardInterrupt:  # 捕获Ctrl+C中断
            logging.info("监控服务已停止")  # 记录信息日志

def main():
    """主函数"""
    try:
        monitor = DesktopMonitor()  # 创建监控器实例
        
        monitor.start_monitoring()  # 开始监控
        
    except Exception as e:  # 捕获启动过程中的异常
        logging.critical(f"程序启动失败: {e}")  # 记录严重错误日志

if __name__ == "__main__":  # 如果作为主程序运行
    main()  # 执行主函数