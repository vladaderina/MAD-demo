import yaml
import logging
from pathlib import Path
from typing import Dict
from datetime import datetime, timezone
import pandas as pd  # Добавлен импорт pandas
from .victoriametrics import VictoriaMetricsClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MetricsCollector:
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.vm_client = VictoriaMetricsClient(self.config["victoriametrics"]["url"])
        self.output_dir = Path(self.config["data"]["output_path"]).parent
        self.output_dir.mkdir(exist_ok=True)

    @staticmethod
    def _load_config(path: str) -> Dict:
        with open(path, "r") as f:
            return yaml.safe_load(f)

    def run(self):
        for metric in self.config["victoriametrics"]["promql_queries"]:
            self._process_metric(metric)

    def _process_metric(self, metric: Dict):
        logger.info(f"Processing metric: {metric['name']}")
        
        time_settings = metric["time_settings"]
        if "fixed_range" in time_settings:
            fr = time_settings["fixed_range"]
            series_data = self.vm_client.query_range(
                metric["query"], fr["start"], fr["end"], fr.get("step", "1m")
            )
        else:
            ar = time_settings["auto_range"]
            end_time = datetime.now(timezone.utc)
            start_time = end_time - pd.to_timedelta(ar["lookback_period"])
            series_data = self.vm_client.query_range(
                metric["query"],
                start_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                end_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                ar["step"]
            )

        for df, labels in series_data:
            self._save_metric(df, metric["name"], labels)

    def _save_metric(self, df: pd.DataFrame, name: str, labels: Dict):
        filename = f"{name}__{'__'.join(f'{k}={v}' for k, v in labels.items())}.csv"
        output_path = self.output_dir / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)  # Создаем директорию если нет
        df.to_csv(output_path, index=False)
        logger.info(f"Saved metric data to {output_path}")

if __name__ == "__main__":
    collector = MetricsCollector("config/config.yaml")  # Уточнен путь к конфигу
    collector.run()