#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import time
from pathlib import Path

import requests


# ============================================================
# 配置
# ============================================================

MIRROR_BASE = "https://hf-mirror.com"

REPO_ID = "Dangindev/VinDR-CXR-VQA"
FILENAME = "data_v1.json"
REVISION = "main"

SAVE_DIR = Path("/home/yuqing/Datasets/VinDr-CXR-VQA")
OUTPUT_PATH = SAVE_DIR / FILENAME
TEMP_PATH = SAVE_DIR / f"{FILENAME}.part"

DOWNLOAD_URL = (
    f"{MIRROR_BASE}/datasets/"
    f"{REPO_ID}/resolve/{REVISION}/{FILENAME}"
)

CHUNK_SIZE = 1024 * 1024  # 每次读取 1 MB
MAX_RETRIES = 10
CONNECT_TIMEOUT = 30
READ_TIMEOUT = 120


def format_size(num_bytes: int) -> str:
    """将字节数格式化为易读形式。"""
    value = float(num_bytes)

    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024:
            return f"{value:.2f} {unit}"
        value /= 1024

    return f"{value:.2f} PB"


def validate_json(path: Path) -> None:
    """验证下载结果是否为完整、合法的 JSON。"""
    print("\n正在校验 JSON 文件……")

    if not path.exists():
        raise FileNotFoundError(f"文件不存在：{path}")

    if path.stat().st_size == 0:
        raise RuntimeError("下载文件为空。")

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    print("JSON 格式校验通过。")
    print(f"文件路径：{path}")
    print(f"文件大小：{format_size(path.stat().st_size)}")
    print(f"顶层类型：{type(data).__name__}")

    if hasattr(data, "__len__"):
        print(f"顶层元素数量：{len(data)}")

    if isinstance(data, list) and data:
        first_item = data[0]
        if isinstance(first_item, dict):
            print(f"首条样本字段：{list(first_item.keys())}")

    elif isinstance(data, dict):
        print(f"顶层字段：{list(data.keys())[:20]}")


def download_file() -> None:
    """从国内镜像下载文件，支持断点续传。"""
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    if OUTPUT_PATH.exists():
        try:
            validate_json(OUTPUT_PATH)
            print("\n目标文件已经存在且格式正常，无需重复下载。")
            return
        except Exception as error:
            print(f"\n现有文件校验失败，将重新下载：{error}")
            OUTPUT_PATH.unlink(missing_ok=True)

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0 VinDr-CXR-VQA-Downloader/1.0",
            "Accept": "*/*",
            "Connection": "keep-alive",
        }
    )

    for attempt in range(1, MAX_RETRIES + 1):
        existing_size = TEMP_PATH.stat().st_size if TEMP_PATH.exists() else 0

        headers = {}
        if existing_size > 0:
            headers["Range"] = f"bytes={existing_size}-"
            print(
                f"\n检测到临时文件，将从 "
                f"{format_size(existing_size)} 处继续下载。"
            )

        try:
            print(f"\n第 {attempt}/{MAX_RETRIES} 次尝试")
            print(f"下载地址：{DOWNLOAD_URL}")
            print(f"保存位置：{OUTPUT_PATH}")

            response = session.get(
                DOWNLOAD_URL,
                headers=headers,
                stream=True,
                allow_redirects=True,
                timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
            )

            # 206 表示服务器接受断点续传。
            if response.status_code == 206:
                write_mode = "ab"
                downloaded_size = existing_size

            # 200 表示完整返回。
            elif response.status_code == 200:
                write_mode = "wb"
                downloaded_size = 0

                # 请求了 Range 但服务器不支持续传，则重新写入。
                if existing_size > 0:
                    print("镜像未接受断点续传，将从头重新下载。")

            else:
                error_preview = response.text[:500]
                raise RuntimeError(
                    f"HTTP 状态码：{response.status_code}\n"
                    f"响应内容：{error_preview}"
                )

            remaining_size = int(response.headers.get("Content-Length", 0))
            total_size = downloaded_size + remaining_size

            last_print_time = 0.0

            with TEMP_PATH.open(write_mode) as file:
                for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                    if not chunk:
                        continue

                    file.write(chunk)
                    downloaded_size += len(chunk)

                    current_time = time.time()
                    if current_time - last_print_time >= 1:
                        if total_size > 0:
                            percentage = downloaded_size / total_size * 100
                            print(
                                "\r进度："
                                f"{format_size(downloaded_size)} / "
                                f"{format_size(total_size)} "
                                f"({percentage:.2f}%)",
                                end="",
                                flush=True,
                            )
                        else:
                            print(
                                f"\r已下载：{format_size(downloaded_size)}",
                                end="",
                                flush=True,
                            )

                        last_print_time = current_time

            print()

            if total_size > 0 and downloaded_size < total_size:
                raise RuntimeError(
                    "下载大小不完整："
                    f"实际 {downloaded_size} bytes，"
                    f"预期 {total_size} bytes。"
                )

            # 先校验临时文件，成功后再正式改名。
            validate_json(TEMP_PATH)
            TEMP_PATH.replace(OUTPUT_PATH)

            print("\n下载成功！")
            print(f"最终文件：{OUTPUT_PATH}")
            return

        except (
            requests.RequestException,
            RuntimeError,
            OSError,
            json.JSONDecodeError,
        ) as error:
            print(f"\n本次下载失败：{error}")

            if attempt >= MAX_RETRIES:
                raise RuntimeError(
                    f"连续尝试 {MAX_RETRIES} 次仍然失败。\n"
                    "请检查服务器能否访问 hf-mirror.com，"
                    "或是否存在失效的代理设置。"
                ) from error

            wait_seconds = min(attempt * 3, 30)
            print(f"{wait_seconds} 秒后重试……")
            time.sleep(wait_seconds)


if __name__ == "__main__":
    try:
        download_file()
    except KeyboardInterrupt:
        print(
            "\n下载已手动停止，临时文件会保留，"
            "下次运行可以继续下载。"
        )
    except Exception as exc:
        print(f"\n程序执行失败：{exc}")
        raise
