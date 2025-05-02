# llm-throughput-eval

Evaluate llm's generation speed via API

## Quick Start

### Get the repository

Clone:

```shell
git clone https://github.com/Orion-zhen/llm-throughput-eval.git
```

Install dependencies:

```shell
pip install -r requirements.txt
```

### Go

```shell
python evel.py -m <your model name> -u <target API URL> -n <total requests> -c <concurrencies> -t <API token>
```

Example:

```shell
python eval.py -m "qwq:32b"
```

This will send 16 requests with 4 concurrency to local ollama API (should be <http://127.0.0.1:11434>) with model qwq:32b

Full arguments:

```shell
usage: eval.py [-h] [--concurrency CONCURRENCY] [--requests REQUESTS] [--url URL] [--model MODEL]

Async HTTP Benchmark Tool

options:
  -h, --help            show this help message and exit
  --concurrency, -c CONCURRENCY
                        Maximum concurrent requests.
  --requests, -n REQUESTS
                        Total number of requests to send.
  --url, -u URL         Base URL for the API (e.g., http://localhost:8000). '/v1/chat/completions' will be appended.
  --model, -m MODEL     Model name to use in the request payload.
  --token, -t TOKEN     Bearer token for API authentication.
```

## Credits

The code is inspired by [this article](https://blog.csdn.net/arkohut/article/details/139076652)
