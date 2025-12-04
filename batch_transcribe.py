import os
import sys
from dotenv import load_dotenv  # 引入库
from groq import Groq
from moviepy import VideoFileClip

# ================= 配置区域 =================
# 1. 加载 .env 文件中的环境变量
load_dotenv()

# 2. 从环境变量获取 API Key (不再硬编码)
API_KEY = os.getenv("GROQ_API_KEY")

# 增加一个安全检查，防止用户忘记配置
if not API_KEY:
    print("❌ 错误: 未找到 API Key。")
    print("请确保你创建了 .env 文件，并设置了 GROQ_API_KEY=你的密钥")
    sys.exit(1)

# 你的视频文件夹路径
INPUT_FOLDER = r"./videos"
# 转录结果保存路径
OUTPUT_FOLDER = r"./transcripts"
# 支持的视频格式后缀
VIDEO_EXTENSIONS = ('.mp4', '.mov', '.avi', '.mkv', '.flv')
# 使用的模型
MODEL_ID = "whisper-large-v3"
# ===========================================

# 初始化 Groq 客户端
client = Groq(api_key=API_KEY)


# ... (后续 extract_audio 和 transcribe_audio_file 函数代码保持不变) ...
# ... (main 函数代码保持不变) ...

# 为了完整性，这里补全 extract_audio 之后的代码结构，确保你可以直接复制
def extract_audio(video_path, audio_path):
    """从视频中提取音频并保存为临时 MP3 文件"""
    try:
        # 使用 moviepy 提取音频
        with VideoFileClip(video_path) as video:
            if video.audio is not None:
                # 降低比特率以减小文件体积，64k 对于语音转录足够了
                video.audio.write_audiofile(audio_path, bitrate="64k", logger=None)
                return True
            else:
                print(f"⚠️ 跳过：文件 {os.path.basename(video_path)} 没有音轨")
                return False
    except Exception as e:
        print(f"❌ 提取音频失败: {video_path}, 错误: {e}")
        return False


def transcribe_audio_file(audio_path):
    """调用 Groq API 转录音频"""
    try:
        with open(audio_path, "rb") as file:
            transcription = client.audio.transcriptions.create(
                file=(os.path.basename(audio_path), file.read()),
                model=MODEL_ID,
                response_format="json",
                language=None
            )
        return transcription.text
    except Exception as e:
        print(f"❌ API 调用失败: {e}")
        return None


def main():
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    # 确保输入目录存在，如果不存在提示用户创建
    if not os.path.exists(INPUT_FOLDER):
        os.makedirs(INPUT_FOLDER)
        print(f"ℹ️ 已创建输入文件夹: {INPUT_FOLDER}，请放入视频文件后重试。")
        return

    for root, dirs, files in os.walk(INPUT_FOLDER):
        for filename in files:
            if filename.lower().endswith(VIDEO_EXTENSIONS):
                video_path = os.path.join(root, filename)
                base_name = os.path.splitext(filename)[0]
                txt_output_path = os.path.join(OUTPUT_FOLDER, f"{base_name}.txt")

                if os.path.exists(txt_output_path):
                    print(f"⏭️ 跳过已存在: {filename}")
                    continue

                print(f"🚀 正在处理: {filename} ...")
                temp_audio_path = "temp_audio_extract.mp3"

                if extract_audio(video_path, temp_audio_path):
                    print(f"   正在转录中 (使用 {MODEL_ID})...")
                    text = transcribe_audio_file(temp_audio_path)

                    if text:
                        with open(txt_output_path, "w", encoding="utf-8") as f:
                            f.write(text)
                        print(f"✅ 完成！已保存至: {base_name}.txt")

                    if os.path.exists(temp_audio_path):
                        os.remove(temp_audio_path)
                print("-" * 30)


if __name__ == "__main__":
    main()