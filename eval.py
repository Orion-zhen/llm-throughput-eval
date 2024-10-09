import aiohttp
import asyncio
import time
import argparse
from tqdm import tqdm


async def fetch(session, url, model: str = "gpt-3.5-turbo"):
    """
    参数:
        session (aiohttp.ClientSession): 用于请求的会话。
        url (str): 要发送请求的 URL。

    返回:
        tuple: 包含完成 token 数量和请求时间。
    """
    start_time = time.time()

    # 固定请求的内容
    json_payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Why is the sky blue?"}],
        "stream": False,
        "temperature": 0.5
    }
    async with session.post(url, json=json_payload) as response:
        response_json = await response.json()
        end_time = time.time()
        request_time = end_time - start_time
        completion_tokens = response_json["usage"][
            "completion_tokens"
        ]  # 从返回的参数里获取生成的 token 的数量
        return completion_tokens, request_time


async def bound_fetch(sem, session, pbar, url, model):
    # 使用信号量 sem 来限制并发请求的数量，确保不会超过最大并发请求数
    async with sem:
        result = await fetch(session, url, model)
        pbar.update(1)
        return result


async def run(load_url, model, max_concurrent_requests, total_requests):
    """
    通过发送多个并发请求来运行基准测试。

    参数:
        load_url (str): 要发送请求的URL。
        max_concurrent_requests (int): 最大并发请求数。
        total_requests (int): 要发送的总请求数。

    返回:
        tuple: 包含完成 token 总数列表和响应时间列表。
    """
    # 创建 Semaphore 来限制并发请求的数量
    sem = asyncio.Semaphore(max_concurrent_requests)

    # 创建一个异步的HTTP会话
    async with aiohttp.ClientSession() as session:
        tasks = []

        # 创建一个进度条来可视化请求的进度
        with tqdm(total=total_requests) as pbar:
            # 循环创建任务，直到达到总请求数
            for _ in range(total_requests):
                # 为每个请求创建一个任务，确保它遵守信号量的限制
                task = asyncio.ensure_future(bound_fetch(sem, session, pbar, load_url, model))
                tasks.append(task)  # 将任务添加到任务列表中

            # 等待所有任务完成并收集它们的结果
            results = await asyncio.gather(*tasks)

        # 计算所有结果中的完成token总数
        completion_tokens = sum(result[0] for result in results)

        # 从所有结果中提取响应时间
        response_times = [result[1] for result in results]

        # 返回完成token的总数和响应时间的列表
        return completion_tokens, response_times


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--concurrency", "-c", type=int, default=4)
    parser.add_argument("--requests", "-r", type=int, default=16)
    parser.add_argument("--url", "-u", type=str, default="http://127.0.0.1:11451", help="no v1 in url")
    parser.add_argument("--model", "-m", type=str, default="gpt-3.5-turbo")
    args = parser.parse_args()

    # vllm 和 ollama 都兼容了 openai 的 api 让测试变得更简单了
    if args.url == "ollama":
        url = "http://127.0.0.1:11434/v1/chat/completions"
    else:
        url = f"{args.url}/v1/chat/completions"

    start_time = time.time()
    completion_tokens, response_times = asyncio.run(run(url, args.model, args.concurrency, args.requests))
    end_time = time.time()

    # 计算总时间
    total_time = end_time - start_time
    # 计算每个请求的平均时间
    avg_time_per_request = sum(response_times) / len(response_times)
    # 计算每秒生成的 token 数量
    tokens_per_second = completion_tokens / total_time

    print(f"Performance Results:")
    print(f"  Total requests            : {args.requests}")
    print(f"  Max concurrent requests   : {args.concurrency}")
    print(f"  Total time                : {total_time:.2f} seconds")
    print(f"  Average time per request  : {avg_time_per_request:.2f} seconds")
    print(f"  Tokens per second         : {tokens_per_second:.2f}")
