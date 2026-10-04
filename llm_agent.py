# ============================================================
# llm_agent.py
# 修改记录：
# 2026-10-03
# - Task 开始阶段的目标图分析 / relevance 继续使用云端 qwen3-vl-plus。
# - 每个 Step 的双图动作决策改为本地 Qwen3-VL-8B-Instruct-FP8。
# - 本地模型通过 http://127.0.0.1:8001/v1 提供 OpenAI 兼容接口。
# - 保留详细耗时日志，用于直接比较公网 API 和本地模型速度。
# - 根据图片扩展名自动生成正确的 data:image/... MIME 类型。
# ============================================================

import os
import time
import base64
import mimetypes
from openai import OpenAI

TARGET_ANALYSIS_MODEL = "qwen3-vl-plus"
ACTION_DECISION_MODEL = "qwen3-vl-4b"
TEXT_MODEL = "qwen-plus-latest"

qwen_client = OpenAI(
    api_key="sk-ws-H.PLRHLML.KI2a.MEYCIQCrvKvUsNLhgZ0fFnYbh__TL7fbxXEBWwrxeTMM5IxENgIhAMeODlFAtEWJ1-m03CK5higmNuHwMVvhyGx3iKlUcjI0",
    base_url=(
        "https://ws-xodt2i0ugny2gftz.cn-beijing.maas.aliyuncs.com/"
        "compatible-mode/v1"
    ),
)

local_client = OpenAI(
    base_url="http://127.0.0.1:8001/v1",
    api_key="LOCAL",
    timeout=120.0,
)

def encode_image(image_path):
    total_start = time.perf_counter()

    file_size_mb = 0.0
    try:
        file_size_mb = os.path.getsize(image_path) / (1024 * 1024)
    except OSError:
        pass

    read_start = time.perf_counter()
    with open(image_path, "rb") as image_file:
        image_bytes = image_file.read()
    read_time = time.perf_counter() - read_start

    encode_start = time.perf_counter()
    base64_image = base64.b64encode(image_bytes).decode("utf-8")
    encode_time = time.perf_counter() - encode_start

    total_time = time.perf_counter() - total_start
    base64_size_mb = len(base64_image.encode("utf-8")) / (1024 * 1024)

    print(
        f"[LLM DETAIL] image={image_path} | "
        f"file={file_size_mb:.2f}MB | "
        f"base64={base64_size_mb:.2f}MB | "
        f"read={read_time:.3f}s | "
        f"encode={encode_time:.3f}s | "
        f"total={total_time:.3f}s"
    )

    return base64_image

def get_image_mime_type(image_path):
    mime_type, _ = mimetypes.guess_type(image_path)

    if mime_type and mime_type.startswith("image/"):
        return mime_type

    ext = os.path.splitext(image_path)[1].lower()
    if ext == ".png":
        return "image/png"
    if ext in (".jpg", ".jpeg"):
        return "image/jpeg"
    if ext == ".webp":
        return "image/webp"

    return "image/jpeg"

def make_data_url(image_path):
    base64_image = encode_image(image_path)
    mime_type = get_image_mime_type(image_path)
    return f"data:{mime_type};base64,{base64_image}"

def get_response(messages, model_name):
    request_start = time.perf_counter()

    api_client = local_client if model_name == ACTION_DECISION_MODEL else qwen_client

    print(
        f"[LLM DETAIL] get_response API request start | "
        f"model={model_name}"
    )

    kwargs = {
        "model": model_name,
        "messages": messages,
    }

    if model_name == ACTION_DECISION_MODEL:
        kwargs["max_tokens"] = 16
        kwargs["temperature"] = 0

    completion = api_client.chat.completions.create(**kwargs)

    request_time = time.perf_counter() - request_start

    print(
        f"[LLM DETAIL] get_response API request finished = "
        f"{request_time:.2f}s"
    )

    return completion.model_dump()

def chat_with_llm(prompt, image_paths=None):
    function_start = time.perf_counter()

    if image_paths:
        model_name = TARGET_ANALYSIS_MODEL

        prepare_start = time.perf_counter()

        message_content = [
            {"type": "text", "text": prompt}
        ]

        image_start = time.perf_counter()
        image_data_url = make_data_url(image_paths)

        print(
            f"[LLM DETAIL] single image prepared in "
            f"{time.perf_counter() - image_start:.3f}s"
        )

        message_content.append(
            {
                "type": "image_url",
                "image_url": {"url": image_data_url},
            }
        )

        messages = [
            {"role": "user", "content": message_content}
        ]

        print(
            f"[LLM DETAIL] message prepared = "
            f"{time.perf_counter() - prepare_start:.3f}s"
        )

    else:
        model_name = TEXT_MODEL
        messages = [
            {"role": "user", "content": prompt}
        ]

    api_start = time.perf_counter()

    print(
        f"[LLM DETAIL] CLOUD API request start | model={model_name}"
    )

    response = qwen_client.chat.completions.create(
        model=model_name,
        messages=messages,
    )

    api_time = time.perf_counter() - api_start

    print(
        f"[LLM DETAIL] CLOUD API request finished = {api_time:.2f}s"
    )

    parse_start = time.perf_counter()
    text = response.choices[0].message.content
    parse_time = time.perf_counter() - parse_start

    print(
        f"[LLM DETAIL] response parse = {parse_time:.4f}s"
    )

    if getattr(response, "usage", None) is not None:
        print("[LLM DETAIL] usage:", response.usage)

    print(
        f"[LLM DETAIL] chat_with_llm total = "
        f"{time.perf_counter() - function_start:.2f}s"
    )

    return text

def chat_with_llm_images(prompt, image_paths=None):
    function_start = time.perf_counter()

    if image_paths:
        prepare_start = time.perf_counter()

        message_content = [
            {"type": "text", "text": prompt}
        ]

        for index, image_path in enumerate(image_paths, start=1):
            image_start = time.perf_counter()

            image_data_url = make_data_url(image_path)

            message_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": image_data_url},
                }
            )

            print(
                f"[LLM DETAIL] image {index}/{len(image_paths)} "
                f"prepared in "
                f"{time.perf_counter() - image_start:.3f}s"
            )

        messages = [
            {"role": "user", "content": message_content}
        ]

        print(
            f"[LLM DETAIL] all images + messages prepared = "
            f"{time.perf_counter() - prepare_start:.3f}s"
        )

    else:
        messages = [
            {"role": "user", "content": prompt}
        ]

    model_name = ACTION_DECISION_MODEL
    api_start = time.perf_counter()

    print(
        f"[LLM DETAIL] LOCAL API request start | model={model_name}"
    )

    response = local_client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_tokens=16,
        temperature=0,
    )

    api_time = time.perf_counter() - api_start

    print(
        f"[LLM DETAIL] LOCAL API request finished = {api_time:.2f}s"
    )

    parse_start = time.perf_counter()
    text = response.choices[0].message.content
    parse_time = time.perf_counter() - parse_start

    print(
        f"[LLM DETAIL] response parse = {parse_time:.4f}s"
    )

    if getattr(response, "usage", None) is not None:
        print("[LLM DETAIL] usage:", response.usage)

    print(
        f"[LLM DETAIL] chat_with_llm_images total = "
        f"{time.perf_counter() - function_start:.2f}s"
    )

    return text

if __name__ == "__main__":
    print("TARGET_ANALYSIS_MODEL =", TARGET_ANALYSIS_MODEL)
    print("ACTION_DECISION_MODEL =", ACTION_DECISION_MODEL)
    print("TEXT_MODEL =", TEXT_MODEL)
    print("LOCAL BASE URL = http://127.0.0.1:8001/v1")
