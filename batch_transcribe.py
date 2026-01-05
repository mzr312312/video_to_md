import os
import sys
import math
import time
import re
import json
from dotenv import load_dotenv
from groq import Groq
from moviepy import VideoFileClip
from pydub import AudioSegment

# ================= 配置区域 =================
load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY")

# 【手动指定 FFmpeg 路径】如果环境变量失效，请取消下面两行的注释并修改路径
# AudioSegment.converter = r"C:\ffmpeg\bin\ffmpeg.exe"
# AudioSegment.ffprobe   = r"C:\ffmpeg\bin\ffprobe.exe"

if not API_KEY:
    print("❌ 错误: 未找到 API Key。请检查 .env 文件。")
    sys.exit(1)

INPUT_FOLDER = r"./videos"
OUTPUT_FOLDER = r"./transcripts"
VIDEO_EXTENSIONS = ('.mp4', '.mov', '.avi', '.mkv', '.flv')
MODEL_ID = "whisper-large-v3"

# 切割设置：每段 15 分钟
CHUNK_DURATION_MS = 15 * 60 * 1000
# ===========================================

client = Groq(api_key=API_KEY)


def extract_audio(video_path, audio_path):
    """从视频中提取音频"""
    try:
        with VideoFileClip(video_path) as video:
            if video.audio is not None:
                video.audio.write_audiofile(audio_path, bitrate="64k", logger=None)
                return True
            else:
                print(f"⚠️ 跳过：文件 {os.path.basename(video_path)} 没有音轨")
                return False
    except Exception as e:
        print(f"❌ 提取音频失败: {e}")
        return False


def transcribe_chunks_with_resume(audio_path, cache_path):
    """带断点续传和自动重试的转录逻辑"""
    combined_chunks = []

    # 加载缓存进度
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            combined_chunks = json.load(f)
        print(f"⏳ 检测到缓存，已跳过前 {len(combined_chunks)} 个已完成片段")

    try:
        audio = AudioSegment.from_file(audio_path)
        duration_ms = len(audio)
        total_chunks = math.ceil(duration_ms / CHUNK_DURATION_MS)

        for i in range(len(combined_chunks), total_chunks):
            start = i * CHUNK_DURATION_MS
            end = min((i + 1) * CHUNK_DURATION_MS, duration_ms)

            chunk_name = f"temp_chunk_{i}.mp3"
            audio[start:end].export(chunk_name, format="mp3", bitrate="64k")

            # API 请求重试逻辑
            while True:
                try:
                    print(f"   进度: {i + 1}/{total_chunks} 正在向 Groq 请求转录...")
                    with open(chunk_name, "rb") as f:
                        transcription = client.audio.transcriptions.create(
                            file=(chunk_name, f.read()),
                            model=MODEL_ID,
                            language="zh"
                        )
                        combined_chunks.append(transcription.text)

                    # 成功后立即更新缓存
                    with open(cache_path, "w", encoding="utf-8") as f:
                        json.dump(combined_chunks, f, ensure_ascii=False)
                    break

                except Exception as e:
                    err_msg = str(e)
                    if "429" in err_msg:
                        # 解析等待时间
                        wait_time = 60
                        match = re.search(r"try again in (\d+m)?([\d\.]+)s", err_msg)
                        if match:
                            m = match.group(1)
                            s = float(match.group(2))
                            wait_time = (int(m[:-1]) * 60 if m else 0) + s + 2

                        print(f"⏳ 触发额度限制，自动暂停 {wait_time} 秒后重试...")
                        time.sleep(wait_time)
                    else:
                        print(f"❌ API 致命错误: {e}")
                        return None

            if os.path.exists(chunk_name):
                os.remove(chunk_name)

        return "\n".join(combined_chunks)

    except Exception as e:
        print(f"❌ 处理音频流时出错: {e}")
        return None


def main():
    if not os.path.exists(OUTPUT_FOLDER): os.makedirs(OUTPUT_FOLDER)
    if not os.path.exists(INPUT_FOLDER):
        os.makedirs(INPUT_FOLDER)
        print(f"ℹ️ 已创建输入文件夹: {INPUT_FOLDER}")
        return

    for filename in os.listdir(INPUT_FOLDER):
        if filename.lower().endswith(VIDEO_EXTENSIONS):
            video_path = os.path.join(INPUT_FOLDER, filename)
            base_name = os.path.splitext(filename)[0]
            txt_output_path = os.path.join(OUTPUT_FOLDER, f"{base_name}.txt")
            cache_path = os.path.join(OUTPUT_FOLDER, f"{base_name}.cache.json")

            if os.path.exists(txt_output_path):
                print(f"⏭️ 跳过已存在结果: {filename}")
                continue

            print(f"🚀 开始处理: {filename}")
            temp_audio = f"temp_{base_name}.mp3"

            if extract_audio(video_path, temp_audio):
                final_text = transcribe_chunks_with_resume(temp_audio, cache_path)

                if final_text:
                    with open(txt_output_path, "w", encoding="utf-8") as f:
                        f.write(final_text)
                    print(f"✅ 全片转录完成！已保存。")

                    # 完成后清理临时文件
                    if os.path.exists(cache_path): os.remove(cache_path)
                    if os.path.exists(temp_audio): os.remove(temp_audio)
                else:
                    print(f"⚠️ {filename} 处理中断，进度已保存至 .cache 文件")

            print("-" * 40)


if __name__ == "__main__":
    main()