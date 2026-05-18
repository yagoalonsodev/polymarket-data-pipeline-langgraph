## Arquitectura

Requisitos cubiertos:

- **(5)** Polymarket API → ETL extracción → DataLake (S3) → Sensor Airflow → Transformaciones → DataWarehouse
- **(6)** DataLake en **Amazon S3**, datos **raw/crudos**, formato **Delta Lake**, incluye timestamp extracción
- **(7)** **Sensor** en Airflow que espera a la llegada del raw antes de transformar/cargar
- **(10)** DW relacional: **NeonDB (Postgres)**
- **(13,14)** Agente AI con **LangGraph** (`StateGraph`: sql → exec → news → answer) + **DatabaseTool** (SQL contra Neon)
- **Observabilidad:** trazas opcionales en **LangSmith** al usar el chatbot Streamlit (`LANGSMITH_API_KEY` en `.env`)

Flujo:

1. **Extract** (Airflow): consulta `/markets` en `POLYMARKET_BASE_URL`, filtra **ACTIVOS** y keywords (`POLYMARKET_KEYWORDS`).
2. **Raw** (S3): escribe en Delta Lake `s3://$S3_BUCKET/$S3_PREFIX/raw/polymarket_markets_delta` particionado por `_dt` y `_hour`.
3. **Sensor**: espera `s3://$S3_BUCKET/$S3_PREFIX/raw/polymarket_markets_delta/dt=YYYY-MM-DD/hour=HH/_SUCCESS`.
4. **Transform+Load**: lee esa partición y carga tablas en Neon (upserts + snapshots).
5. **Chatbot**: Streamlit llama al agente LangGraph, genera SQL, ejecuta en Neon, responde.

## Tablas DW (Neon)

Definidas en `sql/schema.sql`:

- `polymarket.dim_market`
- `polymarket.fact_market_snapshot`
- `polymarket.dim_outcome`
- `polymarket.fact_outcome_snapshot`

