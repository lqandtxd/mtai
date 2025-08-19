# config.py - 配置文件（也可直接运行启动监控程序）
"""
文件名称：config.py
功能描述：本文件是程序的主要配置文件，包含所有可配置的参数。
设计特点：
    1. 主要是配置项定义
    2. 也可直接运行以启动监控程序（通过导入main.py）
    3. 所有配置项都有详细中文注释
    4. 支持灵活的监控间隔配置

使用方法：
    1. 作为配置文件：被main.py导入使用
    2. 直接运行：python config.py （会启动监控程序）

注意事项：
    - 需要与main.py文件在同一目录
    - 修改配置后需要重启程序生效
"""

# ==================== 全局配置 ====================

# 全局配置字典
# 包含程序运行所需的所有全局参数
GLOBAL_CONFIG = {
    # 监控间隔时间（秒）
    # 程序每隔指定秒数执行一次监控循环
    # 默认值：60秒（1分钟）
    # 可根据需要调整为更短或更长的时间
    # 建议范围：5-300秒
    # 设置为0或负数会导致程序立即连续运行，可能造成CPU占用过高
    "monitor_interval_seconds": 60,
    
    # 截图保存目录
    # 所有识别过程中的截图将保存在此目录下
    # 程序会自动创建该目录（如果不存在）
    # 可以使用相对路径或绝对路径
    "screenshot_dir": "screenshots",
    
    # 日志文件路径
    # 程序运行日志保存的文件路径
    # 使用相对路径，位于当前目录下
    "log_file": "monitor.log",
    
    # OCR语言设置
    # 指定Tesseract OCR引擎使用的语言包
    # 常用值：
    #   'chi_sim'：简体中文
    #   'eng'：英文
    #   'jpn'：日文
    #   'kor'：韩文
    # 可以组合使用，如：'chi_sim+eng'
    # 注意：需要预先安装对应的语言包
    "ocr_lang": "eng",
    
    # 清理任务执行间隔（分钟）
    # 每隔指定分钟数执行一次过期文件清理任务
    # 设置为0表示禁用自动清理
    "cleanup_interval_minutes": 30,
    
    # 文件保留时间（小时）
    # 截图文件保留的最长时间
    # 超过此时间的文件将在清理任务中被删除
    # 设置为0表示永久保留
    "retention_hours": 24,
    
    # 是否启用调试模式
    # True：记录详细的调试信息（适合开发和问题排查）
    # False：只记录警告和错误信息（适合生产环境）
    "debug_mode": True,  # 默认开启调试模式，便于查看OCR结果
    
    # 最大重试次数
    # 当检测失败时，最多重试的次数
    # 设置为0表示不重试
    "max_retry_count": 3,
    
    # 重试间隔时间（秒）
    # 每次重试之间的等待时间
    "retry_interval_seconds": 2,
    
    # 屏幕截图质量
    # 仅对支持质量设置的格式有效（如JPEG）
    # 取值范围：1-100，值越大质量越高，文件越大
    "screenshot_quality": 95,
}

# ==================== 识别配置列表 ====================

# 识别配置列表
# 这是一个包含多个识别任务的列表
# 程序会按顺序执行列表中的每个配置
# 支持无限层级的嵌套检测
RECOGNITION_CONFIGS = [
    {
        # 是否启用此配置
        # True：启用该检测任务
        # False：禁用该检测任务（但仍保留在配置中）
        # 可用于临时关闭某个检测而不删除配置
        "enabled": True,
        
        # 配置名称
        # 用于标识和日志记录
        # 建议使用有意义的名称，便于调试和维护
        "name": "第一层文字检测",
        
        # 检测类型
        # 支持两种检测方式：
        #   "text"：基于OCR的文字识别
        #   "template"：基于图像的模板匹配
        "type": "text",
        
        # 检测区域左上角X坐标（像素）
        # 从屏幕左上角开始计算的水平坐标
        # 坐标系统：左上角为(0,0)，向右X增大，向下Y增大
        "x": 0,
        
        # 检测区域左上角Y坐标（像素）
        # 从屏幕左上角开始计算的垂直坐标
        "y": 0,
        
        # 检测区域宽度（像素）
        # 检测区域的水平尺寸
        "width": 300,
        
        # 检测区域高度（像素）
        # 检测区域的垂直尺寸
        "height": 300,
        
        # 要检测的关键词列表
        # 程序会检查OCR识别出的文字是否包含这些关键词
        # 可以设置多个关键词，满足任意一个即视为匹配成功
        "keywords": ["maplestory"],
        
        # 是否区分大小写
        # True：进行大小写敏感的匹配
        # False：进行大小写不敏感的匹配（推荐）
        "case_sensitive": False,
        
        # OCR语言设置（可选）
        # 如果为空，使用GLOBAL_CONFIG中的全局设置
        # 如果指定，则覆盖全局设置，使用指定的语言包
        # 适用于需要混合识别多种语言的场景
        "ocr_lang": "eng",
        
        # 图像预处理配置
        # 用于提高OCR识别准确率的图像处理参数
        "preprocess": {
            # 是否应用高斯模糊
            # 有助于减少图像噪声，提高识别稳定性
            # 建议：对于有噪点的屏幕，启用此选项
            "blur": True,
            
            # 高斯模糊的核大小
            # 必须是奇数（如3, 5, 7）
            # 值越大，模糊效果越强，但处理时间越长
            "blur_kernel": 3,
            
            # 是否应用阈值处理
            # 将图像转换为黑白二值图，有助于提高OCR准确率
            "threshold": False,
            
            # 阈值处理方法
            # 支持两种方法：
            #   "otsu"：自动计算最佳阈值（推荐）
            #   "fixed"：使用固定的阈值
            "threshold_method": "otsu",
            
            # 固定阈值（仅当threshold_method为"fixed"时有效）
            # 取值范围：0-255
            # 建议值：127（中间值）
            # "threshold_value": 127,
        },
        
        # 回调函数类型
        # 当检测到匹配时执行的操作类型
        # 支持的类型：
        #   "logging"：记录日志
        #   "mouse_click"：执行鼠标点击
        #   "keyboard_input"：执行键盘输入
        #   "custom_script"：执行自定义Python脚本
        "callback_type": "logging",
        
        # 回调函数参数
        # 根据不同的回调类型，参数内容不同
        "callback_params": {
            # 日志消息内容
            # 如果为空，使用默认消息
            "message": "第一层检测成功",
            
            # 是否在日志中包含坐标信息
            # True：在消息后附加检测到的坐标
            # False：只记录纯消息
            "include_coords": True
        },
        
        # X坐标偏移量（像素）
        # 在原始匹配坐标基础上增加的偏移量
        # 用于精确定位实际点击位置
        # 可以为负值（向左偏移）
        "coord_offset_x": 25,
        
        # Y坐标偏移量（像素）
        # 在原始匹配坐标基础上增加的偏移量
        # 用于精确定位实际点击位置
        # 可以为负值（向上偏移）
        "coord_offset_y": 50,
        
        # 嵌套检测配置
        # 在当前检测成功后，继续在匹配区域执行下一层检测
        # 支持无限层级的嵌套，实现复杂的多层验证逻辑
        "nested_detection": {
            # 第二层嵌套检测配置
            "name": "第二层检测",
            "type": "text",
            # 注意：嵌套检测的x,y坐标以父级匹配点为原点
            # 所以这里只设置width和height
            "width": 250,
            "height": 100,
            "keywords": ["forsaken"],
            "case_sensitive": False,
            "preprocess": {
                "threshold": True,
                "threshold_value": 127  # 使用固定阈值127
            },
            "callback_type": "logging",
            "callback_params": {
                "message": "第二层检测成功",
                "include_coords": True
            },
            # "coord_offset_x": 5,  # 再次偏移+5
            # "coord_offset_y": 5,
            
            # # 第三层嵌套检测
            # # 可以继续嵌套，支持无限层级
            # "nested_detection": {
            #     "name": "第三层检测",
            #     "type": "text",
            #     "width": 200,
            #     "height": 80,
            #     "keywords": ["第三层关键词"],
            #     "callback_type": "logging",
            #     "callback_params": {
            #         "message": "第三层检测成功",
            #         "include_coords": True
            #     },
            #     "coord_offset_x": 5,  # 再次偏移+5
            #     "coord_offset_y": 5
            #     # 可以继续添加更多层级...
            # }
        }
    },
    
    # 模板匹配示例配置
    # 展示如何使用图像模板进行匹配
    {
        "enabled": False,
        "name": "模板匹配检测",
        "type": "template",  # 使用模板匹配方式
        "x": 200,
        "y": 200,
        "width": 400,
        "height": 300,
        
        # 模板图片路径
        # 用于模板匹配的参考图片文件路径
        # 图片应该是灰度图或可以转换为灰度图
        # 推荐使用PNG格式，保持图像质量
        "template_path": "template.png",
        
        # 匹配阈值
        # 匹配相似度阈值，范围0.0-1.0
        # 值越大，要求匹配越精确
        # 建议值：0.7-0.9
        # 过高可能导致匹配失败，过低可能导致误匹配
        "match_threshold": 0.8,
        
        # 回调类型
        "callback_type": "logging",
        "callback_params": {
            "message": "模板匹配成功"
        },
        
        # 坐标偏移量
        "coord_offset_x": 10,
        "coord_offset_y": 10,
        
        # 模板匹配后的嵌套检测
        "nested_detection": {
            "name": "模板后文字检测",
            "type": "text",
            "width": 200,
            "height": 100,
            "keywords": ["确认"],
            "callback_type": "logging",
            "callback_params": {
                # x,y是相对于模板匹配中心点的坐标
                "x": 0,  # 在匹配点上点击
                "y": 0,
                "clicks": 1  # 点击次数
            },
            "coord_offset_x": 20,  # 再次偏移
            "coord_offset_y": 20
        }
    },
]

# ==================== 程序入口点 ====================

if __name__ == "__main__":
    """
    当直接运行此文件时，启动监控程序
    通过导入main.py来启动主程序
    这样设计的好处：
        1. config.py主要作为配置文件
        2. 保持配置和逻辑分离
        3. 仍支持直接运行启动程序
    """
    try:
        # 导入主程序模块
        # 注意：确保main.py与本文件在同一目录
        import main
        
        # 启动主程序
        # main.py应该有run()函数作为程序入口
        main.run()
        
    except ImportError as e:
        print(f"导入main.py失败: {e}")
        print("请确保main.py文件存在且在同一目录")
        exit(1)
        
    except Exception as e:
        print(f"启动程序时发生错误: {e}")
        exit(1)