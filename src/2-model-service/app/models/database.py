import psycopg2
from psycopg2 import sql
import json
from typing import Dict, Any, Optional

class DatabaseManager:
    def __init__(self, dsn: str):
        self.dsn = dsn

    def _get_connection(self):
        return psycopg2.connect(self.dsn)

    def save_model(self, name: str, labels: Dict, config: Dict, history: Dict) -> str:
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    sql.SQL("""
                        INSERT INTO modeling.models 
                        (name, labels, config, history)
                        VALUES (%s, %s, %s, %s)
                        RETURNING id
                    """),
                    (name, json.dumps(labels), json.dumps(config), json.dumps(history))
                )
                model_id = cur.fetchone()[0]
                conn.commit()
                return model_id

    def get_model(self, model_id: str) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    sql.SQL("SELECT * FROM modeling.models WHERE id = %s"),
                    (model_id,)
                )
                return cur.fetchone()