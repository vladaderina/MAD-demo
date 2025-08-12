import os
import asyncio
import logging
import argparse
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple

import asyncpg
import aiohttp


class RetrainScheduler:
    """Service for scheduling model retraining for a single model."""
    
    def __init__(
        self,
        db_conn_string: str,
        trainer_api_url: str,
        model_name: str,
        metric_name: str,
        interval: str,
        strategy: str
    ):
        self.db_conn_string = db_conn_string
        self.trainer_api_url = trainer_api_url
        self.model_name = model_name
        self.metric_name = metric_name
        self.interval = self._parse_interval(interval)
        self.strategy = strategy
        self.db_pool = None
        self._setup_logging()

    def _parse_interval(self, interval_str: str) -> timedelta:
        """Convert interval string to timedelta."""
        if interval_str.endswith('d'):
            return timedelta(days=int(interval_str[:-1]))
        elif interval_str.endswith('h'):
            return timedelta(hours=int(interval_str[:-1]))
        raise ValueError(f"Invalid interval format: {interval_str}")

    def _setup_logging(self):
        """Configure logging for this model instance."""
        self.logger = logging.getLogger(f'RetrainScheduler.{self.model_name}')
        self.logger.setLevel(logging.INFO)
        
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            f'[%(levelname)s] %(asctime)s - {self.model_name} - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    async def _get_model_for_retrain(self) -> Optional[dict]:
        """Check if the model needs retraining."""
        async with self.db_pool.acquire() as conn:
            return await conn.fetchrow("""
                SELECT 
                    mi.id as model_id,
                    mi.name as model_name,
                    mi.training_start,
                    mi.training_end,
                    m.id as version_id,
                    m.version
                FROM models_info mi
                JOIN models m ON m.model_id = mi.id AND m.status = 'active'
                WHERE mi.name = $1
                AND (
                    mi.training_end IS NULL OR
                    NOW() >= mi.training_end + $2::interval
                )
            """, self.model_name, f"{self.interval.total_seconds()} seconds")

    async def _update_model_training_period(
        self, 
        model_id: int, 
        new_start: datetime, 
        new_end: datetime
    ) -> None:
        """Update model training period in database."""
        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                UPDATE models_info
                SET training_start = $1, training_end = $2
                WHERE id = $3
            """, new_start, new_end, model_id)

    async def _create_new_model_version(
        self, 
        model_id: int, 
        current_version: str
    ) -> str:
        """Create new model version in database."""
        async with self.db_pool.acquire() as conn:
            # Simple version increment logic
            new_version = f"{float(current_version) + 0.1:.1f}"
            
            # Copy hyperparams from active version
            active_params = await conn.fetchval("""
                SELECT hyperparams FROM models
                WHERE model_id = $1 AND status = 'active'
                ORDER BY created_at DESC LIMIT 1
            """, model_id)
            
            await conn.execute("""
                INSERT INTO models (model_id, status, version, hyperparams)
                VALUES ($1, 'waiting', $2, $3)
            """, model_id, new_version, active_params)
            
            return new_version

    async def _mark_model_for_retraining(
        self, 
        model_id: int, 
        version_id: int
    ) -> None:
        """Mark model version for retraining."""
        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                UPDATE models
                SET status = 'pending_retrain'
                WHERE id = $1
            """, version_id)

    def _calculate_new_training_period(
        self,
        current_start: datetime,
        current_end: datetime,
    ) -> Tuple[datetime, datetime]:
        """Calculate new training period based on strategy."""
        now = datetime.now(timezone.utc)
        
        if self.strategy == 'sliding_window':
            duration = current_end - current_start
            new_end = now
            new_start = new_end - duration
        elif self.strategy == 'expanding_window':
            new_start = current_start
            new_end = now
        else:
            new_start, new_end = current_start, current_end
        
        return new_start, new_end

    async def _notify_trainer_service(
        self,
        model_id: int,
        version_id: int,
        new_version: str
    ) -> bool:
        """Notify trainer service about retraining request."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.trainer_api_url}/api/v1/retrain",
                    json={
                        "model_id": model_id,
                        "version_id": version_id,
                        "new_version": new_version
                    },
                    timeout=10
                ) as resp:
                    if resp.status != 202:
                        raise ValueError(f"API returned status {resp.status}")
                    return True
        except Exception as e:
            self.logger.error(f"Error notifying trainer service: {str(e)}")
            return False

    async def _process_model_retraining(self, model: dict) -> bool:
        """Handle full retraining pipeline for a single model."""
        try:
            # 1. Calculate new training period
            new_start, new_end = self._calculate_new_training_period(
                model['training_start'],
                model['training_end']
            )
            
            # 2. Create new version
            new_version = await self._create_new_model_version(
                model['model_id'],
                model['version']
            )
            
            # 3. Update training period
            await self._update_model_training_period(
                model['model_id'],
                new_start,
                new_end
            )
            
            # 4. Mark for retraining
            await self._mark_model_for_retraining(
                model['model_id'],
                model['version_id']
            )
            
            # 5. Notify trainer service
            success = await self._notify_trainer_service(
                model['model_id'],
                model['version_id'],
                new_version
            )
            
            if success:
                self.logger.info(
                    f"Model prepared for retraining. "
                    f"New period: {new_start} - {new_end}, version: {new_version}"
                )
            return success
            
        except Exception as e:
            self.logger.error(
                f"Error processing model retraining: {str(e)}",
                exc_info=True
            )
            return False

    async def run(self):
        """Execute single retraining check for the configured model."""
        self.db_pool = await asyncpg.create_pool(self.db_conn_string)
        
        try:
            model = await self._get_model_for_retrain()
            if model:
                self.logger.info("Model requires retraining")
                success = await self._process_model_retraining(model)
                self.logger.info(f"Retraining {'succeeded' if success else 'failed'}")
            else:
                self.logger.info("Model does not require retraining")
        except Exception as e:
            self.logger.error(f"Critical error: {str(e)}", exc_info=True)
            raise
        finally:
            if self.db_pool:
                await self.db_pool.close()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Model Retrain Scheduler for single model'
    )
    parser.add_argument(
        '--model-name',
        required=True,
        help='Name of the model to check for retraining'
    )
    parser.add_argument(
        '--metric-name',
        required=True,
        help='Primary metric for this model'
    )
    parser.add_argument(
        '--interval',
        required=True,
        help='Retrain interval (e.g. 7d, 24h)'
    )
    parser.add_argument(
        '--strategy',
        choices=['sliding_window', 'expanding_window'],
        required=True,
        help='Retraining strategy'
    )
    return parser.parse_args()


def main():
    """Entry point for the scheduler."""
    args = parse_args()
    
    db_conn_string = os.getenv("DB_CONN_STRING")
    if not db_conn_string:
        raise ValueError("DB_CONN_STRING environment variable is required")
    
    trainer_api_url = os.getenv("TRAINER_API_URL", "http://model-trainer:8080")
    
    scheduler = RetrainScheduler(
        db_conn_string=db_conn_string,
        trainer_api_url=trainer_api_url,
        model_name=args.model_name,
        metric_name=args.metric_name,
        interval=args.interval,
        strategy=args.strategy
    )
    
    asyncio.run(scheduler.run())


if __name__ == "__main__":
    main()