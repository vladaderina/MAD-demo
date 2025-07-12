import requests
import pandas as pd
import yaml
from datetime import datetime, timezone
import os
import re

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def sanitize_filename(s: str) -> str:
    return re.sub(r'[^\w\-.]', '_', s)

def fetch_metric(base_url, metric_config):
    query = metric_config["query"]
    time_settings = metric_config["time_settings"]

    # Определяем временные параметры
    if "fixed_range" in time_settings:
        fr = time_settings["fixed_range"]
        params = {
            "query": query,
            "start": fr["start"],
            "end": fr["end"],
            "step": fr.get("step", "1m")
        }
    else:
        ar = time_settings["auto_range"]
        end_time = datetime.now(timezone.utc)
        start_time = end_time - pd.to_timedelta(ar["lookback_period"])
        params = {
            "query": query,
            "start": start_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "end": end_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "step": ar["step"]
        }

    try:
        response = requests.get(f"{base_url}/api/v1/query_range", params=params)
        response.raise_for_status()
        data = response.json()

        print(f"[DEBUG] Full response: {data}")

        if data["status"] != "success":
            print(f"Error in response: {data.get('error', 'Unknown error')}")
            return []

        results = data["data"]["result"]
        if not results:
            print("No data returned from the query")
            return []

        dfs = []
        for series in results:
            if "values" in series:
                df = pd.DataFrame(series["values"], columns=["timestamp", "value"])
                labels = series["metric"]
                dfs.append((df, labels))

        return dfs

    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return []
    except KeyError as e:
        print(f"Unexpected response format: {e}")
        return []

def save_metric_data(df: pd.DataFrame, metric_name: str, labels: dict, output_dir: str):
    try:
        os.makedirs(output_dir, exist_ok=True)
        # Составим имя файла из меток
        label_part = "__".join(f"{k}={sanitize_filename(v)}" for k, v in labels.items())
        filename = f"{metric_name}__{label_part}.csv"
        output_path = os.path.join(output_dir, filename)

        # Сохраняем только timestamp и value
        df[["timestamp", "value"]].to_csv(output_path, index=False)
        print(f"[INFO] Saved data to {output_path}")
    except Exception as e:
        print(f"Error saving data: {str(e)}")

def main(config_path="config.yaml"):
    try:
        config = load_config(config_path)
        vm_conf = config["victoriametrics"]
        output_dir = os.path.dirname(config["data"]["output_path"])

        for metric in vm_conf["promql_queries"]:
            name = metric["name"]
            print(f"[INFO] Fetching metric '{name}'...")

            series_list = fetch_metric(
                base_url=vm_conf["url"],
                metric_config=metric
            )

            if not series_list:
                print(f"[WARNING] No data returned for metric '{name}'")
                continue

            for df, labels in series_list:
                save_metric_data(df, name, labels, output_dir)

    except Exception as e:
        print(f"Fatal error: {str(e)}")

if __name__ == "__main__":
    main()
