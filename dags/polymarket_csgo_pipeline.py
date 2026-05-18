"""
Pipeline horario (Opción B) para mercados ACTIVOS de Polymarket filtrados por keywords CSGO.

Arquitectura (cada ejecución del DAG):
Polymarket API -> RAW (Delta en S3) -> Sensor S3 (_SUCCESS) -> Transform + Load -> Neon (Postgres DW)

Tras las transformaciones de la partición correspondiente, los datos se escriben en Neon en la misma corrida
(`transform_and_load_to_neon`), cumpliendo el flujo Data Lake -> Data Warehouse del enunciado.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone

from airflow.providers.standard.operators.python import PythonOperator
from airflow.providers.standard.sensors.python import PythonSensor
from airflow.sdk import DAG

from src.datalake.markers import write_success_marker
from src.datalake.s3_paths import S3Location
from src.datalake.write_raw_delta import write_raw_markets_delta_to_s3
from src.dw.neon import (
    NeonConfig,
    ensure_schema,
    insert_outcome_snapshot,
    insert_snapshot,
    make_engine,
    upsert_market,
    upsert_outcome,
)
from src.polymarket.client import PolymarketClient
from src.polymarket.snapshot import add_snapshot_metadata, snapshot_meta
from src.transform.normalize import RawPartition, read_raw_markets_partition, snapshot_timestamp_from_partition


def _env(name: str, default: str | None = None) -> str | None:
    v = os.environ.get(name)
    if v is None or v == "":
        return default
    return v


def _extract_to_s3_raw(**context) -> dict:
    base_url = _env("POLYMARKET_BASE_URL", "https://gamma-api.polymarket.com") or ""
    keywords = (_env("POLYMARKET_KEYWORDS", "CSGO,Counter-Strike") or "").split(",")
    bucket = _env("S3_BUCKET")
    prefix = _env("S3_PREFIX", "polymarket") or ""
    if not bucket:
        raise ValueError("S3_BUCKET no está configurado")

    aws_region = _env("AWS_REGION")
    aws_access_key_id = _env("AWS_ACCESS_KEY_ID")
    aws_secret_access_key = _env("AWS_SECRET_ACCESS_KEY")
    aws_session_token = _env("AWS_SESSION_TOKEN")

    meta = snapshot_meta()
    client = PolymarketClient(base_url=base_url)

    markets = client.fetch_active_markets_for_keywords(
        keywords=keywords,
        page_limit=100,
        max_pages=101,
    )
    markets = add_snapshot_metadata(markets, meta)

    s3loc = S3Location(bucket=bucket, prefix=prefix)
    delta_uri = s3loc.raw_markets_delta_uri()
    n = write_raw_markets_delta_to_s3(
        delta_table_uri=delta_uri,
        markets=markets,
        aws_region=aws_region,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        aws_session_token=aws_session_token,
        mode="append",
    )

    success_key = s3loc.raw_markets_success_key(dt=meta.dt, hour=meta.hour)
    write_success_marker(
        bucket=bucket,
        key=success_key,
        aws_region=aws_region,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        aws_session_token=aws_session_token,
        body=f"ok rows={n} extracted_at={meta.extracted_at_utc}\n",
    )

    logging.info("RAW escrito: %s (rows=%s) + marker s3://%s/%s", delta_uri, n, bucket, success_key)
    return {
        "delta_uri": delta_uri,
        "bucket": bucket,
        "success_key": success_key,
        "dt": meta.dt,
        "hour": meta.hour,
        "rows": n,
        "extracted_at": meta.extracted_at_utc,
    }

def _transform_and_load_to_neon(**context) -> dict:
    neon_url = _env("NEON_DATABASE_URL")
    if not neon_url:
        raise ValueError("NEON_DATABASE_URL no está configurado")

    aws_region = _env("AWS_REGION")
    aws_access_key_id = _env("AWS_ACCESS_KEY_ID")
    aws_secret_access_key = _env("AWS_SECRET_ACCESS_KEY")
    aws_session_token = _env("AWS_SESSION_TOKEN")

    ti = context["ti"]
    payload = ti.xcom_pull(task_ids="extract_to_s3_raw") or {}
    delta_uri = payload.get("delta_uri")
    dt = payload.get("dt")
    hour = payload.get("hour")
    if not delta_uri or not dt or hour is None:
        raise ValueError("No se pudo resolver delta_uri/dt/hour desde XCom")

    partition = RawPartition(dt=str(dt), hour=int(hour))
    df = read_raw_markets_partition(
        delta_table_uri=str(delta_uri),
        partition=partition,
        aws_region=aws_region,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        aws_session_token=aws_session_token,
    )

    snapshot_ts = snapshot_timestamp_from_partition(partition)
    markets = df.to_dict(orient="records") if not df.empty else []

    schema_path = "/opt/airflow/sql/schema.sql"
    with open(schema_path, "r", encoding="utf-8") as f:
        schema_sql = f.read()

    engine = make_engine(NeonConfig(database_url=str(neon_url)))
    ensure_schema(engine, schema_sql=schema_sql)

    for m in markets:
        upsert_market(engine, market=m)
        insert_snapshot(engine, snapshot_ts=snapshot_ts, market=m)
        # Normalización de outcomes: outcomes + outcomePrices alineados por índice
        market_id = str(m.get("id") or m.get("marketId") or m.get("market_id") or "")
        if not market_id:
            continue
        outcomes = m.get("outcomes")
        prices = m.get("outcomePrices") or m.get("outcome_prices")
        # En RAW suelen venir como strings JSON por compatibilidad con Delta
        import json

        if isinstance(outcomes, str):
            try:
                outcomes = json.loads(outcomes)
            except Exception:  # noqa: BLE001
                outcomes = None
        if isinstance(prices, str):
            try:
                prices = json.loads(prices)
            except Exception:  # noqa: BLE001
                prices = None

        if isinstance(outcomes, list) and isinstance(prices, list):
            for idx, label in enumerate(outcomes):
                outcome_id = upsert_outcome(
                    engine,
                    market_id=market_id,
                    outcome_index=idx,
                    outcome_label=str(label) if label is not None else None,
                )
                prob = None
                try:
                    prob = float(prices[idx]) if idx < len(prices) and prices[idx] is not None else None
                except Exception:  # noqa: BLE001
                    prob = None
                insert_outcome_snapshot(
                    engine,
                    snapshot_ts=snapshot_ts,
                    market_id=market_id,
                    outcome_id=outcome_id,
                    extracted_at=m.get("_extracted_at"),
                    probability=prob,
                    raw={"outcome_label": label, "outcome_price": prices[idx] if idx < len(prices) else None},
                )

    logging.info("DW cargado: markets=%d snapshot_ts=%s", len(markets), snapshot_ts.isoformat())
    return {"loaded_markets": len(markets), "snapshot_ts": snapshot_ts.isoformat()}


with DAG(
    dag_id="polymarket_csgo_hourly_pipeline",
    description="Snapshots horarios Polymarket (ACTIVOS) filtrados por CSGO -> S3 raw delta -> Neon",
    schedule="0 * * * *",
    start_date=datetime(2026, 3, 1, tzinfo=timezone.utc),
    catchup=False,
    default_args={
        "retries": 3,
        "retry_delay": timedelta(minutes=5),
    },
    tags=["polymarket", "csgo", "s3", "neon"],
    doc_md=__doc__,
) as dag:
    extract_to_s3_raw = PythonOperator(
        task_id="extract_to_s3_raw",
        python_callable=_extract_to_s3_raw,
    )

    # Sensor (punto 7): esperar marker en S3 antes de transformaciones
    # Nota: el key exacto lo calcula el task de extracción; aquí lo resolvemos vía XCom.
    def _marker_exists(**context) -> bool:
        import boto3

        aws_region = _env("AWS_REGION")
        aws_access_key_id = _env("AWS_ACCESS_KEY_ID")
        aws_secret_access_key = _env("AWS_SECRET_ACCESS_KEY")
        aws_session_token = _env("AWS_SESSION_TOKEN")

        ti = context["ti"]
        payload = ti.xcom_pull(task_ids="extract_to_s3_raw") or {}
        bucket = payload.get("bucket")
        key = payload.get("success_key")
        if not bucket or not key:
            raise ValueError("No se pudo resolver bucket/key desde XCom")

        session = boto3.session.Session(
            aws_access_key_id=aws_access_key_id or None,
            aws_secret_access_key=aws_secret_access_key or None,
            aws_session_token=aws_session_token or None,
            region_name=aws_region or None,
        )
        s3 = session.client("s3")
        try:
            s3.head_object(Bucket=bucket, Key=key)
            logging.info("Marker encontrado: s3://%s/%s", bucket, key)
            return True
        except Exception as e:  # noqa: BLE001
            logging.info("Marker aún no disponible (%s): s3://%s/%s", type(e).__name__, bucket, key)
            return False

    wait_for_raw_marker = PythonSensor(
        task_id="wait_for_raw_marker",
        python_callable=_marker_exists,
        poke_interval=30,
        timeout=60 * 30,
        mode="poke",
    )

    transform_and_load_to_neon = PythonOperator(
        task_id="transform_and_load_to_neon",
        python_callable=_transform_and_load_to_neon,
    )

    extract_to_s3_raw >> wait_for_raw_marker >> transform_and_load_to_neon

