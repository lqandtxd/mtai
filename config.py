# config.py - 配置文件
import os  # 导入操作系统接口模块，用于文件路径操作

# 全局配置
GLOBAL_CONFIG = {
    "screenshot_dir": "screenshots",  # 截图保存目录名称
    "ocr_lang": "chi_sim+eng",  # OCR识别语言，支持中文简体和英文
    "monitor_interval_minutes": 1,  # 监控检查间隔时间（分钟）
    "cleanup_interval_hours": 1,  # 自动清理过期截图的间隔时间（小时）
    "retention_hours": 24  # 截图文件保留时间（小时）
}

# 识别配置列表，包含多个识别区域的配置
RECOGNITION_CONFIGS = [
    {
        "enabled": False,  # 是否启用此配置项，True为启用，False为禁用
        "name": "login_button",  # 配置名称，用于标识和日志记录
        "type": "template",  # 识别类型：template(模板匹配) 或 text(文字识别)
        "x": 500,  # 检测区域左上角X坐标
        "y": 300,  # 检测区域左上角Y坐标  
        "width": 100,  # 检测区域宽度
        "height": 50,  # 检测区域高度
        "template_path": "templates/login_button.png",  # 模板图片文件路径
        "match_threshold": 0.8,  # 模板匹配阈值，0-1之间，值越高要求匹配越精确
        "callback_type": "mouse_click",  # 匹配成功后的回调动作类型
        "callback_params": {  # 回调动作的参数
            "x": 550,  # 鼠标点击的X坐标
            "y": 325  # 鼠标点击的Y坐标
        }
    },
    {
        "enabled": True,  # 是否启用此配置项
        "name": "error_message",  # 配置名称
        "type": "text",  # 识别类型为文字识别
        "x": 20,  # 检测区域左上角X坐标
        "y": 20,  # 检测区域左上角Y坐标
        "width": 300,  # 检测区域宽度
        "height": 300,  # 检测区域高度
        "keywords": ["twilight"],  # 需要检测的关键词列表
        "case_sensitive": False,  # 是否区分大小写，False表示不区分
        "ocr_lang": "chi_sim+eng",  # 此区域使用的OCR语言
        "callback_type": "logging",  # 回调类型为日志记录
        "callback_params": {  # 回调参数
            "message": "检测到错误信息！"  # 要记录的日志消息
        }
    },
    {
        "enabled": False,  # 是否启用此配置项
        "name": "welcome_text",  # 配置名称
        "type": "text",  # 识别类型为文字识别
        "x": 200,  # 检测区域左上角X坐标
        "y": 200,  # 检测区域左上角Y坐标
        "width": 200,  # 检测区域宽度
        "height": 50,  # 检测区域高度
        "keywords": ["welcome", "欢迎"],  # 需要检测的关键词列表
        "case_sensitive": False,  # 是否区分大小写
        "ocr_lang": "chi_sim+eng",  # 此区域使用的OCR语言
        "callback_type": "keyboard_input",  # 回调类型为键盘输入
        "callback_params": {  # 回调参数
            "text": "hello123"  # 要输入的文本内容
        }
    },
    {
        "enabled": False,  # 此配置被禁用，不会执行
        "name": "test_template",  # 配置名称
        "type": "template",  # 识别类型为模板匹配
        "x": 0,  # 检测区域左上角X坐标
        "y": 0,  # 检测区域左上角Y坐标
        "width": 200,  # 检测区域宽度
        "height": 200,  # 检测区域高度
        "template_path": "templates/test_icon.png",  # 模板图片文件路径
        "match_threshold": 0.9,  # 模板匹配阈值
        "callback_type": "mouse_click",  # 回调类型为鼠标点击
        "callback_params": {  # 回调参数
            "x": 100,  # 鼠标点击的X坐标
            "y": 100  # 鼠标点击的Y坐标
        }
    }
]
# 启动主程序
if __name__ == "__main__":
    import main
    main.main()