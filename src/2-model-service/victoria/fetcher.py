import aiohttp
import pandas as pd

async def fetch_metric_data(metric_name, config):
    # Найди нужный запрос из promql_queries
    for q in config["victoriametrics"]["promql_queries"]:
        if metric_name in q["name"]:
            query = q["query"]
            break
    else:
        raise ValueError(f"No PromQL query for metric {metric_name}")

    # Построй URL и скачай данные
    url = f"{config['victoriametrics']['url']}/api/v1/query_range"
    params = {
        "query": query,
        "start": "...",  # из fixed_range или auto_range
        "end": "...",
        "step": "60"
    }

    async with aiohttp.ClientSession() as session:
        async with session.get(url, params=params) as resp:
            result = await resp.json()

    # Преобразуем в pandas.DataFrame
    values = result["data"]["result"][0]["values"]
    df = pd.DataFrame(values, columns=["timestamp", "value"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
    df["value"] = df["value"].astype(float)
    return df
