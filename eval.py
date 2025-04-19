import time
import aiohttp
import asyncio
import argparse
from typing import NamedTuple, Optional
from tqdm.asyncio import tqdm as async_tqdm


class RequestResult(NamedTuple):
    success: bool
    completion_tokens: Optional[int]
    request_time: float
    error: Optional[Exception]


async def fetch(session: aiohttp.ClientSession, url: str, model: str) -> RequestResult:
    """
    发送一个异步请求到指定的URL。

    参数:
        session (aiohttp.ClientSession): 用于请求的会话。
        url (str): 要发送请求的 URL。
        model (str): 模型名称。

    返回:
        RequestResult: 包含请求结果、耗时和可能的错误信息。
    """
    start_time = time.time()
    completion_tokens = None
    error = None
    success = False

    json_payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Why is the sky blue?"}],
        "stream": False,
        "temperature": 0.5,
    }

    # 为请求设置超时
    timeout = aiohttp.ClientTimeout(total=3600)

    try:
        async with session.post(url, json=json_payload, timeout=timeout) as response:
            # 检查HTTP状态码
            if response.status != 200:
                response_text = await response.text()
                raise aiohttp.ClientResponseError(
                    request_info=response.request_info,
                    history=response.history,
                    status=response.status,
                    message=f"HTTP error {response.status}: {response_text}",
                    headers=response.headers,
                )

            response_json = await response.json()

            # 从返回的参数里获取生成的 token 的数量
            assert isinstance(response_json, dict)
            if not hasattr(response_json, "usage"):
                raise KeyError("Missing 'usage' in response")
            completion_tokens = response_json.get("usage", {}).get("completion_tokens")
            if completion_tokens is None:
                raise KeyError("Missing 'usage' or 'completion_tokens' in response")

            success = True

    except (aiohttp.ClientError, asyncio.TimeoutError, KeyError, Exception) as e:
        # 捕获各种可能的错误：网络错误, 超时, JSON结构错误
        error = e

    finally:
        end_time = time.time()
        request_time = end_time - start_time
        # 即使请求失败也要记录时间

    return RequestResult(
        success=success,
        completion_tokens=completion_tokens,
        request_time=request_time,
        error=error,
    )


async def bounded_fetch(
    sem: asyncio.Semaphore,
    session: aiohttp.ClientSession,
    pbar: async_tqdm,
    url: str,
    model: str,
) -> RequestResult:
    """
    使用信号量sem来限制并发请求的数量，确保不会超过最大并发请求数，并调用fetch。
    同时更新进度条。
    """
    async with sem:
        # 在信号量内执行请求，确保并发限制
        result = await fetch(session, url, model)
        # 无论成功或失败，都更新进度条表示一个任务完成
        pbar.update(1)
        return result


async def run(
    load_url: str, model: str, max_concurrent_requests: int, total_requests: int
) -> tuple[list[RequestResult], float]:
    """
    通过发送多个并发请求来运行基准测试。

    参数:
        load_url (str): 要发送请求的URL。
        model (str): 模型名称。
        max_concurrent_requests (int): 最大并发请求数。
        total_requests (int): 要发送的总请求数。

    返回:
        tuple: 包含 RequestResult 列表和总耗时。
    """
    # 创建 Semaphore 来限制并发请求的数量
    sem = asyncio.Semaphore(max_concurrent_requests)

    # 创建一个异步的HTTP会话
    # Connector 可以配置连接限制等，这里使用默认，但可按需配置
    async with aiohttp.ClientSession() as session:
        tasks: list[asyncio.Task[RequestResult]] = []

        with async_tqdm(total=total_requests, desc="Running requests") as pbar:
            # 循环创建任务，直到达到总请求数
            for _ in range(total_requests):
                # 为每个请求创建一个任务，确保它遵守信号量的限制
                task = asyncio.create_task(
                    bounded_fetch(sem, session, pbar, load_url, model)
                )
                tasks.append(task)  # 将任务添加到任务列表中

            # 等待所有任务完成并收集它们的结果
            results: list[RequestResult] = await asyncio.gather(*tasks)

    # run函数本身不负责计算最终指标，只负责执行和收集原始结果。
    # 指标计算放在调用者 (main 块) 中进行，职责更分离。
    return results, time.time()  # 返回原始结果列表和运行结束时间


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Async HTTP Benchmark Tool")
    parser.add_argument(
        "--concurrency", "-c", type=int, default=4, help="Maximum concurrent requests."
    )
    parser.add_argument(
        "--requests",
        "-n",
        type=int,
        default=16,
        help="Total number of requests to send.",
    )
    parser.add_argument(
        "--url",
        "-u",
        type=str,
        default="http://127.0.0.1:11434",
        help="Base URL for the API (e.g., http://localhost:8000). '/v1/chat/completions' will be appended.",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="gpt-3.5-turbo",
        help="Model name to use in the request payload.",
    )
    args = parser.parse_args()

    full_url = f"{args.url.rstrip('/')}/v1/chat/completions"  # 确保没有双斜杠

    print(f"Starting benchmark for {full_url} with model '{args.model}'...")

    start_time = time.time()
    # 运行异步主函数并获取结果列表和结束时间
    all_results, end_async_time = asyncio.run(
        run(full_url, args.model, args.concurrency, args.requests)
    )
    # 计算总的执行时间，从脚本开始到 asyncio.run 结束
    total_process_time = time.time() - start_time

    # --- 处理结果和计算指标 ---
    successful_results = [res for res in all_results if res.success]
    failed_results = [res for res in all_results if not res.success]

    total_completion_tokens = sum(
        res.completion_tokens
        for res in successful_results
        if res.completion_tokens is not None
    )
    successful_response_times = [res.request_time for res in successful_results]

    print("\n--- Performance Results ---")
    print(f"  Total requests sent     : {args.requests}")
    print(f"  Requests succeeded      : {len(successful_results)}")
    print(f"  Requests failed         : {len(failed_results)}")
    print(f"  Max concurrent requests : {args.concurrency}")
    print(f"  Total execution time    : {total_process_time:.2f} seconds")

    if successful_results:
        # 计算总的成功请求时间（所有成功请求的总和，不是并发时间）
        total_successful_request_time = sum(successful_response_times)
        # 计算平均时间
        avg_time_per_successful_request = total_successful_request_time / len(
            successful_results
        )

        print(f"  Total completion tokens : {total_completion_tokens}")
        # tokens per second 通常是 based on total_completion_tokens / total_execution_time
        tokens_per_second = (
            total_completion_tokens / total_process_time
            if total_completion_tokens > 0 and total_process_time > 0
            else 0
        )

        print(
            f"  Average time/successful : {avg_time_per_successful_request:.2f} seconds"
        )
        print(
            f"  Tokens per second       : {tokens_per_second:.2f}"
        )  # 这是一个衡量吞吐量的好指标

    else:
        print("  No requests succeeded.")

    if failed_results:
        print("\n--- Failed Request Details ---")
        # 打印一些失败请求的详情，帮助调试
        for i, res in enumerate(
            failed_results[:5]
        ):  # 只打印前5个失败详情，避免输出过多
            print(
                f"  Failed request {i+1}: Error Type: {type(res.error).__name__}, Message: {res.error}"
            )
        if len(failed_results) > 5:
            print(f"  ... and {len(failed_results) - 5} more failures.")
