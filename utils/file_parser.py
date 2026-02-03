# utils/file_parser.py
import os
import fitz  # PyMuPDF
import logging
import numpy as np
import cv2

# 尝试导入 PaddleOCR，如果没装则优雅降级
try:
    from paddleocr import PaddleOCR

    HAS_OCR = True
except ImportError:
    HAS_OCR = False

logger = logging.getLogger("JobAgent")


class FileParser:
    _ocr_instance = None

    @classmethod
    def get_ocr_engine(cls):
        """单例模式获取 OCR 引擎，避免重复加载模型"""
        if not HAS_OCR:
            logger.warning("⚠️ 未检测到 PaddleOCR 库，将无法处理图片或扫描版PDF。")
            return None

        if cls._ocr_instance is None:
            logger.info("🔄 正在初始化 OCR 引擎 (首次运行会自动下载模型)...")
            # use_angle_cls=True 自动纠正方向, lang="ch" 支持中英文
            # show_log=False 关闭繁杂的日志
            cls._ocr_instance = PaddleOCR(use_angle_cls=True, lang="ch", show_log=False)
        return cls._ocr_instance

    @staticmethod
    def _pixmap_to_tensor(pix):
        """
        核心辅助方法：将 PyMuPDF 的 Pixmap 转换为 PaddleOCR 可用的 Numpy 数组 (OpenCV 格式)
        """
        # 1. 从 pixmap 获取原始像素数据
        # pix.samples 是一个 bytes 对象
        # pix.h, pix.w, pix.n 分别是高、宽、通道数

        # 2. 转换为 numpy 数组
        img_array = np.frombuffer(pix.samples, dtype=np.uint8)

        # 3. 重塑形状 (Height, Width, Channels)
        img_array = img_array.reshape(pix.h, pix.w, pix.n)

        # 4. 颜色空间转换
        # PyMuPDF 默认通常是 RGB (或 RGBA)，OpenCV/PaddleOCR 内部处理通常也支持 RGB
        # 但为了稳健，如果含有 Alpha 通道 (RGBA)，通常需要转为 RGB
        if pix.n == 4:  # RGBA -> RGB
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
        elif pix.n == 3:  # RGB (保持不变，或者根据需要转 BGR)
            # PaddleOCR 实际上接收 RGB 即可
            pass

        return img_array

    @staticmethod
    def _ocr_image_array(img_array):
        """内部方法：对图片数组(Numpy)进行 OCR"""
        ocr = FileParser.get_ocr_engine()
        if not ocr:
            return "[系统提示: OCR组件未安装，无法读取图片内容]"

        try:
            # ocr.ocr 接收 numpy array
            result = ocr.ocr(img_array, cls=True)
            text_result = []

            # result 结构通常是 [ [ [box], [text, score] ], ... ]
            # 注意：如果图片里没字，result 可能为 [None] 或 None
            if result and result[0]:
                for line in result[0]:
                    text_content = line[1][0]
                    # 可以在这里过滤掉置信度太低的内容，比如 score < 0.5
                    text_result.append(text_content)

            return "\n".join(text_result)
        except Exception as e:
            logger.error(f"OCR 识别出错: {e}")
            return f"[OCR 错误: {e}]"

    @staticmethod
    def parse_pdf(file_path):
        """处理 PDF：智能混合模式 (文本提取 + 扫描件OCR)"""
        text_content = ""
        try:
            doc = fitz.open(file_path)
            for page_num, page in enumerate(doc):
                logger.info(f"📄 正在解析 PDF 第 {page_num + 1} 页...")

                # 1. 尝试直接提取文本
                page_text = page.get_text()

                # 2. 质量检测逻辑
                # 如果提取出的有效字符很少(比如<100)，且装了OCR，则认为是扫描件
                if len(page_text.strip()) < 100 and HAS_OCR:
                    logger.info(f"   -> 第 {page_num + 1} 页检测为扫描/图片页，启动 OCR...")

                    # 3. 渲染页面为图片
                    # matrix=fitz.Matrix(2, 2) 表示放大 2 倍，显著提升 OCR 准确率
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))

                    # 4. 格式转换 (Pixmap -> Numpy)
                    img_array = FileParser._pixmap_to_tensor(pix)

                    # 5. 执行 OCR
                    ocr_text = FileParser._ocr_image_array(img_array)

                    # 标记一下这是 OCR 出来的
                    page_text = f"[OCR 识别内容]\n{ocr_text}"

                elif len(page_text.strip()) < 50 and not HAS_OCR:
                    page_text = "[该页看似扫描件，但系统未安装 PaddleOCR，无法识别]"

                text_content += f"\n--- Page {page_num + 1} ---\n{page_text}\n"

            doc.close()
            return text_content
        except Exception as e:
            logger.error(f"PDF 解析失败: {e}", exc_info=True)
            return f"PDF 读取错误: {str(e)}"

    @staticmethod
    def parse_image(file_path):
        """处理图片文件：直接 OCR"""
        if not HAS_OCR:
            return "系统未安装 PaddleOCR，无法识别图片简历。"

        try:
            # PaddleOCR 可以直接传文件路径，也可以传 numpy array
            # 这里直接传路径最简单
            ocr = FileParser.get_ocr_engine()
            result = ocr.ocr(file_path, cls=True)

            text_result = []
            if result and result[0]:
                for line in result[0]:
                    text_result.append(line[1][0])
            return "\n".join(text_result)
        except Exception as e:
            logger.error(f"图片解析失败: {e}")
            return f"图片读取错误: {str(e)}"

    @staticmethod
    def read_file(file_path):
        """统一入口"""
        if not os.path.exists(file_path):
            return "错误：文件不存在"

        ext = os.path.splitext(file_path)[1].lower()

        logger.info(f"📂 开始读取文件: {file_path}")

        if ext == ".pdf":
            return FileParser.parse_pdf(file_path)
        elif ext in [".jpg", ".jpeg", ".png", ".bmp"]:
            return FileParser.parse_image(file_path)
        elif ext in [".txt", ".md"]:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        else:
            return f"不支持的文件格式: {ext}"