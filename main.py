# main.py - 主程序文件
"""
文件名称：main.py
功能描述：本文件是程序的主逻辑文件，包含所有核心功能的实现。
设计特点：
    1. 包含程序的主要逻辑和功能实现
    2. 依赖config.py中的配置
    3. 实现了可配置的监控间隔
    4. 所有关键代码都有详细中文注释

使用方法：
    python main.py
    或通过运行config.py启动

注意事项：
    - 需要与config.py文件在同一目录
    - 需要安装必要的依赖库
"""

# 导入必要的标准库
import time
import os
import sys
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import shutil

# 导入第三方库
# 注意：这些库需要提前安装
# 安装命令：pip install pytesseract pillow opencv-python pyautogui
try:
    import pytesseract
    from PIL import Image, ImageGrab
    import cv2
    import numpy as np
    import pyautogui
except ImportError as e:
    print(f"缺少必要的依赖库: {e}")
    print("请运行: pip install pytesseract pillow opencv-python pyautogui")
    sys.exit(1)

# 导入配置文件
# config.py应该与本文件在同一目录
try:
    import config
except ImportError:
    print("无法导入config.py，请确保文件存在且在同一目录")
    sys.exit(1)

# ==================== 日志系统初始化 ====================

def setup_logging():
    """初始化日志系统"""
    # 创建日志格式
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    
    # 配置根日志记录器
    logging.basicConfig(
        level=logging.DEBUG if config.GLOBAL_CONFIG["debug_mode"] else logging.INFO,
        format=log_format,
        datefmt=date_format,
        handlers=[
            # 文件处理器：将日志写入文件
            logging.FileHandler(config.GLOBAL_CONFIG["log_file"], encoding='utf-8'),
            # 控制台处理器：将日志输出到控制台
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logging.info("日志系统初始化完成")
    logging.debug(f"调试模式: {config.GLOBAL_CONFIG['debug_mode']}")

# ==================== 屏幕截图功能 ====================

def capture_screen(region: Dict[str, int] = None) -> Optional[Image.Image]:
    """
    截取屏幕指定区域
    
    参数:
        region: 区域字典，包含x, y, width, height
                如果为None，则截取全屏
    
    返回:
        PIL Image对象，截取的屏幕图像
        如果失败返回None
    """
    try:
        if region is None:
            # 截取全屏
            screenshot = ImageGrab.grab()
        else:
            # 截取指定区域
            box = (
                region["x"],
                region["y"],
                region["x"] + region["width"],
                region["y"] + region["height"]
            )
            screenshot = ImageGrab.grab(bbox=box)
        
        # 保存截图（如果需要）
        if config.GLOBAL_CONFIG["screenshot_dir"]:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"screenshot_{timestamp}.png"
            filepath = os.path.join(config.GLOBAL_CONFIG["screenshot_dir"], filename)
            
            # 确保目录存在
            os.makedirs(config.GLOBAL_CONFIG["screenshot_dir"], exist_ok=True)
            
            # 保存截图
            screenshot.save(filepath, quality=config.GLOBAL_CONFIG["screenshot_quality"])
            logging.debug(f"截图已保存: {filename}")
        
        return screenshot
        
    except Exception as e:
        logging.error(f"截图失败: {e}")
        return None

# ==================== 图像预处理功能 ====================

def preprocess_image(image: Image.Image, preprocess_config: Dict[str, Any]) -> np.ndarray:
    """
    对图像进行预处理，提高OCR识别准确率
    
    参数:
        image: PIL Image对象
        preprocess_config: 预处理配置字典
    
    返回:
        处理后的OpenCV图像（numpy数组）
    """
    try:
        # 转换为OpenCV格式（BGR）
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # 转换为灰度图
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        
        # 使用CLAHE增强对比度
        if preprocess_config.get("enhance_contrast", False):
            clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)

        # 应用高斯模糊
        if preprocess_config.get("blur", False):
            kernel_size = preprocess_config.get("blur_kernel", 3)
            # 确保核大小为奇数
            kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
            gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
        
        # 应用阈值处理
        if preprocess_config.get("threshold", False):
            method = preprocess_config.get("threshold_method", "otsu")
            
            if method == "otsu":
                _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)  # 注意：加了 INV
                gray = binary
            elif method == "fixed":
                threshold_value = preprocess_config.get("threshold_value", 127)
                _, binary = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY_INV)  # 加了 INV
                gray = binary

        # 保存截图（如果需要）
        if config.GLOBAL_CONFIG["screenshot_dir"]:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"screenshot_{timestamp}.png"
            filepath = os.path.join(config.GLOBAL_CONFIG["screenshot_dir"], filename)
            # 确保目录存在
            os.makedirs(config.GLOBAL_CONFIG["screenshot_dir"], exist_ok=True)
            success = cv2.imwrite(filepath, gray)
            if success:
                logging.debug(f"已保存预处理后的图像: {filepath}")
            else:
                logging.warning(f"无法保存预处理图像，路径可能无效或磁盘错误: {filepath}")
        else:
            logging.warning("未指定 filepath，无法保存处理后的图像。")

        return gray
        
    except Exception as e:
        logging.error(f"图像预处理失败: {e}")
        return np.array(image)

# ==================== OCR文字识别 ====================

def ocr_recognition(image: np.ndarray, lang: str = None) -> str:
    """
    使用Tesseract OCR识别图像中的文字
    
    参数:
        image: OpenCV图像（numpy数组）
        lang: OCR语言代码，如'chi_sim', 'eng'等
              如果为None，使用全局配置
    
    返回:
        识别出的文字字符串
    """
    try:
        # 确定使用的语言
        if not lang:
            lang = config.GLOBAL_CONFIG["ocr_lang"]
        
        # 使用pytesseract进行OCR识别
        text = pytesseract.image_to_string(image, lang=lang)
        
        # 清理识别结果
        # 去除首尾空白字符
        text = text.strip()
        
        # 替换多余的空白字符为单个空格
        import re
        text = re.sub(r'\s+', ' ', text)
        
        return text
        
    except Exception as e:
        logging.error(f"OCR识别失败: {e}")
        return ""

# ==================== 模板匹配 ====================

def template_match(screen_image: np.ndarray, template_path: str, threshold: float) -> Optional[Dict[str, int]]:
    """
    在屏幕图像中匹配模板图片
    
    参数:
        screen_image: 屏幕图像（OpenCV格式）
        template_path: 模板图片路径
        threshold: 匹配阈值（0.0-1.0）
    
    返回:
        匹配位置字典，包含'x'和'y'坐标
        如果未找到匹配，返回None
    """
    try:
        # 检查模板文件是否存在
        if not os.path.exists(template_path):
            logging.error(f"模板文件不存在: {template_path}")
            return None
        
        # 读取模板图像
        template = cv2.imread(template_path, 0)  # 以灰度模式读取
        if template is None:
            logging.error(f"无法读取模板文件: {template_path}")
            return None
        
        # 获取模板尺寸
        h, w = template.shape
        
        # 执行模板匹配
        result = cv2.matchTemplate(screen_image, template, cv2.TM_CCOEFF_NORMED)
        
        # 找到最佳匹配位置
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        # 检查匹配度是否达到阈值
        if max_val >= threshold:
            # 返回匹配中心点坐标
            center_x = max_loc[0] + w // 2
            center_y = max_loc[1] + h // 2
            return {"x": center_x, "y": center_y}
        else:
            logging.debug(f"模板匹配失败，相似度: {max_val:.3f} < 阈值: {threshold}")
            return None
            
    except Exception as e:
        logging.error(f"模板匹配失败: {e}")
        return None

# ==================== 回调函数执行 ====================

def execute_callback(config: Dict[str, Any], match_position: Dict[str, int] = None):
    """
    执行回调函数
    
    参数:
        config: 包含回调配置的字典
        match_position: 匹配到的位置坐标（用于鼠标点击等操作）
    """
    callback_type = config["callback_type"]
    params = config["callback_params"]
    
    # 获取实际执行位置
    exec_x, exec_y = get_execution_position(config, match_position)
    
    try:
        # 根据回调类型执行不同操作
        if callback_type == "logging":
            message = params.get("message", f"检测成功: {config['name']}")
            if params.get("include_coords", False) and match_position:
                message += f" (坐标: {exec_x}, {exec_y})"
            logging.info(message)
            
        elif callback_type == "mouse_click":
            # 执行鼠标点击
            clicks = params.get("clicks", 1)
            interval = params.get("interval", 0.1)
            button = params.get("button", "left")  # left, right, middle
            
            # 移动鼠标并点击
            pyautogui.moveTo(exec_x, exec_y)
            pyautogui.click(clicks=clicks, interval=interval, button=button)
            
            logging.info(f"执行鼠标点击: ({exec_x}, {exec_y}), {clicks}次, {button}键")
            
        elif callback_type == "keyboard_input":
            # 执行键盘输入
            text = params.get("text", "")
            pause = params.get("pause", 0.1)
            
            pyautogui.typewrite(text, interval=pause)
            logging.info(f"执行键盘输入: {text}")
            
        elif callback_type == "custom_script":
            # 执行自定义Python脚本
            script_path = params.get("script_path", "")
            if os.path.exists(script_path):
                try:
                    # 动态执行脚本
                    with open(script_path, 'r', encoding='utf-8') as f:
                        script_code = f.read()
                    exec(script_code, {"config": config, "position": match_position})
                    logging.info(f"执行自定义脚本成功: {script_path}")
                except Exception as e:
                    logging.error(f"执行自定义脚本失败 {script_path}: {e}")
            else:
                logging.error(f"自定义脚本文件不存在: {script_path}")
                
        else:
            logging.warning(f"未知的回调类型: {callback_type}")
            
    except Exception as e:
        logging.error(f"执行回调失败 {config['name']}: {e}")

def get_execution_position(config: Dict[str, Any], match_position: Dict[str, int] = None) -> tuple:
    """
    计算实际执行位置
    
    参数:
        config: 配置字典
        match_position: 匹配到的位置
    
    返回:
        (x, y) 坐标元组
    """
    if match_position is None:
        # 使用配置中的区域中心点
        x = config["x"] + config["width"] // 2
        y = config["y"] + config["height"] // 2
    else:
        # 使用匹配到的位置
        x = match_position["x"]
        y = match_position["y"]
    
    # 应用坐标偏移
    x += config.get("coord_offset_x", 0)
    y += config.get("coord_offset_y", 0)
    
    return x, y

# ==================== 单个检测任务执行 ====================

def execute_detection(config: Dict[str, Any], level: int = 1) -> bool:
    """
    执行单个检测任务
    
    参数:
        config: 检测配置字典
        level: 嵌套层级（用于日志显示）
    
    返回:
        bool: 检测是否成功
    """
    # 生成缩进，用于日志中的层级显示
    indent = "  " * (level - 1)
    
    try:
        # 记录开始检测日志
        logging.info(f"{indent}开始检测: {config['name']} (层级 {level})")
        
        # 根据检测类型执行不同操作
        if config["type"] == "text":
            # 文字识别检测
            success = text_detection(config, level)
        elif config["type"] == "template":
            # 模板匹配检测
            success = template_detection(config, level)
        else:
            logging.error(f"{indent}未知的检测类型: {config['type']}")
            return False
            
        if success:
            logging.info(f"{indent}✓ 检测成功: {config['name']}")
            return True
        else:
            logging.warning(f"{indent}✗ 检测失败: {config['name']}")
            return False
            
    except Exception as e:
        logging.error(f"{indent}检测过程中发生异常 {config['name']}: {e}", exc_info=True)
        return False

def text_detection(config: Dict[str, Any], level: int) -> bool:
    """
    执行文字识别检测
    
    参数:
        config: 文字检测配置
        level: 层级
    
    返回:
        是否检测成功
    """
    indent = "  " * (level - 1)
    
    # 截取指定区域
    region = {
        "x": config["x"],
        "y": config["y"],
        "width": config["width"],
        "height": config["height"]
    }
    
    screenshot = capture_screen(region)
    if screenshot is None:
        logging.error(f"{indent}截图失败")
        return False
    
    # 图像预处理
    processed_image = preprocess_image(screenshot, config["preprocess"])
    
    # 确定OCR语言
    ocr_lang = config.get("ocr_lang") or config.GLOBAL_CONFIG["ocr_lang"]
    
    # OCR识别
    recognized_text = ocr_recognition(processed_image, ocr_lang)
    
    # === 关键修正 ===
    # 无论匹配结果如何，都记录OCR识别结果
    # 这样即使检测失败，我们也能看到识别出了什么
    logging.debug(f"{indent}OCR识别结果: '{recognized_text}'")
    
    # 关键词匹配
    keywords = config["keywords"]
    case_sensitive = config["case_sensitive"]
    
    # 如果识别结果为空，直接返回失败
    if not recognized_text:
        logging.debug(f"{indent}OCR识别结果为空，检测失败")
        return False
    
    # 根据是否区分大小写进行匹配
    text_to_search = recognized_text if case_sensitive else recognized_text.lower()
    keyword_list = keywords if case_sensitive else [k.lower() for k in keywords]
    
    # 检查是否包含任意关键词
    matched = any(keyword in text_to_search for keyword in keyword_list)
    
    if matched:
        # 计算匹配位置（区域中心）
        match_position = {
            "x": config["x"] + config["width"] // 2,
            "y": config["y"] + config["height"] // 2
        }
        
        # 执行回调
        execute_callback(config, match_position)
        
        # 执行嵌套检测
        if "nested_detection" in config:
            nested_config = config["nested_detection"]
            if nested_config.get("enabled", True):
                # 递归调用嵌套检测
                nested_success = execute_detection(nested_config, level + 1)
                if not nested_success:
                    return False
        
        return True
    else:
        # 即使匹配失败，我们也记录了OCR结果，便于调试
        logging.debug(f"{indent}未找到关键词，检测失败")
        return False

def template_detection(config: Dict[str, Any], level: int) -> bool:
    """
    执行模板匹配检测
    
    参数:
        config: 模板检测配置
        level: 层级
    
    返回:
        是否检测成功
    """
    indent = "  " * (level - 1)
    
    # 截取全屏用于模板匹配
    full_screenshot = capture_screen()
    if full_screenshot is None:
        logging.error(f"{indent}全屏截图失败")
        return False
    
    # 转换为灰度图
    gray_screen = cv2.cvtColor(np.array(full_screenshot), cv2.COLOR_RGB2GRAY)
    
    # 执行模板匹配
    match_result = template_match(
        gray_screen,
        config["template_path"],
        config["match_threshold"]
    )
    
    if match_result is None:
        return False
    
    # 执行回调
    execute_callback(config, match_result)
    
    # 执行嵌套检测
    if "nested_detection" in config:
        nested_config = config["nested_detection"]
        if nested_config.get("enabled", True):
            # 为嵌套检测设置正确的区域（以匹配点为中心）
            nested_config["x"] = match_result["x"] - nested_config["width"] // 2
            nested_config["y"] = match_result["y"] - nested_config["height"] // 2
            
            nested_success = execute_detection(nested_config, level + 1)
            if not nested_success:
                return False
    
    return True

# ==================== 文件清理功能 ====================

def cleanup_old_files():
    """清理过期的截图文件"""
    # 获取当前时间
    now = datetime.now()
    # 计算保留时间阈值
    retention = timedelta(hours=config.GLOBAL_CONFIG["retention_hours"])
    
    # 检查截图目录是否存在
    screenshot_dir = config.GLOBAL_CONFIG["screenshot_dir"]
    if not os.path.exists(screenshot_dir):
        os.makedirs(screenshot_dir, exist_ok=True)
        logging.debug(f"截图目录不存在: {screenshot_dir}；生成目录")
        return
    
    # 遍历目录中的所有文件
    try:
        for filename in os.listdir(screenshot_dir):
            filepath = os.path.join(screenshot_dir, filename)
            # 确保是文件（不是目录）
            if os.path.isfile(filepath):
                # 获取文件创建时间
                file_time = datetime.fromtimestamp(os.path.getctime(filepath))
                # 检查是否超过保留时间
                if now - file_time > retention:
                    try:
                        os.remove(filepath)
                        logging.info(f"已删除过期文件: {filename}")
                    except Exception as e:
                        logging.error(f"删除文件失败 {filename}: {e}")
    except Exception as e:
        logging.error(f"清理文件时发生错误: {e}")

# ==================== 主监控循环 ====================

def monitor_loop():
    """主监控循环"""
    logging.info("监控程序启动")
    logging.info(f"配置项: {len(config.RECOGNITION_CONFIGS)} 个检测任务")
    logging.info(f"监控间隔: {config.GLOBAL_CONFIG['monitor_interval_seconds']} 秒")
    
    # 记录上次执行清理任务的时间
    last_cleanup = datetime.now()
    
    try:
        while True:
            logging.info("=" * 50)
            logging.info("开始新一轮监控循环")
            
            # 执行所有启用的检测配置
            for detection_config in config.RECOGNITION_CONFIGS:
                if detection_config["enabled"]:
                    # 执行单个检测任务
                    execute_detection(detection_config)
            
            # 执行清理任务（按配置间隔）
            current_time = datetime.now()
            cleanup_interval = timedelta(minutes=config.GLOBAL_CONFIG["cleanup_interval_minutes"])
            
            # 检查是否需要执行清理
            if (config.GLOBAL_CONFIG["cleanup_interval_minutes"] > 0 and 
                current_time - last_cleanup > cleanup_interval):
                cleanup_old_files()
                last_cleanup = current_time
            
            # 等待下一次循环
            interval = config.GLOBAL_CONFIG["monitor_interval_seconds"]
            logging.debug(f"等待 {interval} 秒后执行下一次监控...")
            time.sleep(interval)
            
    except KeyboardInterrupt:
        logging.info("监控程序被用户中断 (Ctrl+C)")
    except Exception as e:
        logging.error(f"监控程序发生未预期的异常: {e}", exc_info=True)
    finally:
        logging.info("监控程序结束")


def clear_previous_screenshots(screenshot_dir):
    """
    清除指定目录下的所有截图文件。
    
    参数:
        screenshot_dir: 存储截图的目录路径
    """
    if os.path.exists(screenshot_dir):
        for filename in os.listdir(screenshot_dir):
            file_path = os.path.join(screenshot_dir, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                logging.error(f'Failed to delete {file_path}. Reason: {e}')
    else:
        logging.debug(f"截图目录不存在: {screenshot_dir}")

# ==================== 程序入口函数 ====================

def run():
    """
    程序入口函数
    这是程序的主入口点
    """
    screenshot_dir = config.GLOBAL_CONFIG["screenshot_dir"]
    clear_previous_screenshots(screenshot_dir)
    # 1. 初始化日志系统
    setup_logging()
    
    # 2. 启动主监控循环
    monitor_loop()
    
    # 3. 程序正常退出
    logging.info("程序退出")
    sys.exit(0)

# ==================== 程序入口点 ====================

if __name__ == "__main__":
    """
    当直接运行main.py时的入口点
    """
    run()