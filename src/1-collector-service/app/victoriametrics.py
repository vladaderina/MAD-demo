import requests
import pandas as pd
from datetime import datetime, timezone
import os
import re
import logging
from typing import List, Dict, Tuple, Optional

logger = logging.getLogger(__name__)

class VictoriaMetricsClient:
    def __init__(self, base_url: str):
        self.base_url = base_url

    @staticmethod
    def sanitize_filename(s: str) -> str:
        return re.sub(r'[^\w\-.]', '_', s)

    def query_range(self, query: str, start: str, end: str, step: str) -> List[Tuple[pd.DataFrame, dict]]:
        try:
            params = {"query": query, "start": start, "end": end, "step": step}
            response = requests.get(f"{self.base_url}/api/v1/query_range", params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            if data["status"] != "success":
                logger.error(f"Query failed: {data.get('error', 'Unknown error')}")
                return []

            results = data["data"]["result"]
            if not results:
                logger.warning("No data returned from query")
                return []

            return self._process_results(results)
        except Exception as e:
            logger.error(f"Query error: {e}")
            return []

    def _process_results(self, results: List[dict]) -> List[Tuple[pd.DataFrame, dict]]:
        processed = []
        for series in results:
            if "values" in series:
                df = pd.DataFrame(series["values"], columns=["timestamp", "value"])
                labels = series["metric"]
                processed.append((df, labels))
        return processed