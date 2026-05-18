# Proyecto Final: Data Engineering + AI Agents con Polymarket (CSGO)

Este proyecto implementa:

- Pipeline de datos orquestado con **Apache Airflow**
- **DataLake en S3** con datos **RAW** en **Delta Lake**
- **Sensor en Airflow** que espera la llegada del RAW antes de transformar
- Transformaciones y carga a **Data Warehouse relacional (NeonDB/Postgres)**
- **Agente LangGraph** + **Chatbot Streamlit** para consultas en lenguaje natural (SQL contra Neon)

## Requisitos (alto nivel)

- Docker (para Airflow)
- Credenciales AWS (S3 real) y un bucket
- Un proyecto en NeonDB (Postgres) y su `NEON_DATABASE_URL`
- Clave de OpenAI si vas a usar el agente con `langchain-openai`

## Setup rápido

1. Copia variables:

```bash
cp .env.example .env
```

2. Crea el esquema del data warehouse en Neon (una vez por proyecto; si no, la base queda sin tablas y el pipeline no puede cargar datos):

```bash
pip install 'psycopg[binary]'   # si aún no lo tienes
python scripts/init_neon_schema.py
```

El script lee `NEON_DATABASE_URL` desde `.env` y ejecuta `sql/schema.sql` (schema `polymarket` + tablas).

3. Arranca Airflow:

```bash
docker compose up -d
```

4. UI Airflow: `http://localhost:8080`

- Usuario: `yago`
- Password: `yago`

## Ejecutar el chatbot (Streamlit)

Desde tu máquina (no dentro de Airflow):

```bash
export $(cat .env | xargs)  # o carga variables a tu manera
streamlit run streamlit_app/app.py
```

## LangSmith

Hay **dos cosas distintas** en smith.langchain.com:

| Qué | Para qué | Cómo |
|-----|----------|------|
| **Tracing / Runs** | Ver trazas del chatbot | Solo Streamlit + variables en `.env` |
| **Studio** (grafo interactivo) | Probar el agente en el navegador | **Primero** `langgraph dev` en tu Mac |

Si abres **Studio** sin el servidor local, verás *Failed to initialize Studio* / *Failed to fetch*.

### Variables en `.env`

```env
LANGSMITH_API_KEY=lsv2_pt_...
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=proyecto3-polymarket-csgo
```

### A) Trazas (sin Studio)

```bash
streamlit run streamlit_app/app.py
```

Luego: [smith.langchain.com](https://smith.langchain.com) → proyecto → **Tracing**.

### B) LangGraph Studio (grafo)

1. Instala el CLI (una vez):

```bash
uv sync --extra studio
```

2. En la carpeta del proyecto, **deja esta terminal abierta**:

```bash
langgraph dev
```

Debe mostrar algo como `API: http://127.0.0.1:2024` y `Studio UI: https://smith.langchain.com/studio/?baseUrl=...`.

3. Abre el enlace de **Studio UI** que imprime el comando (o Chrome):

`https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024`

4. Prueba en **Graph** (`Question` o JSON) o en **Chat** (escribe la pregunta en el hilo).

```json
{"question": "¿Qué mercados son los más activos actualmente?"}
```

Tras reiniciar `langgraph dev`, la pestaña **Chat** queda activa (el estado debe extender `MessagesState`, no solo añadir un campo `messages` suelto). Si sigue en gris, recarga la página con hard refresh (Cmd+Shift+R).

**Safari:** suele bloquear `http://127.0.0.1:2024` desde la web HTTPS → usa **Chrome** o `langgraph dev --tunnel` / ngrok en el puerto 2024.

## Preguntas de demo (preparadas)

Úsalas en **Streamlit** o en **LangGraph Studio → Chat**. Funcionan mejor si el DAG ya ha cargado datos en Neon (tablas `polymarket.*` con filas) y `NEON_DATABASE_URL` está en `.env`.

### Enunciado (Parte 2) — recomendadas para la presentación

| Pregunta | Qué hace el agente |
|----------|-------------------|
| ¿Qué mercados son los más activos actualmente? | Mercados `active=true` ordenados por `updated_at` (actividad reciente en Polymarket). |
| Top 10 mercados con más volumen en activo | Top 10 por **volumen acumulado** del último snapshot, solo mercados activos (partidos CSGO con volumen real). |
| ¿Qué mercado cambió más de probabilidad en las últimas 24 horas? | Δ probabilidad por outcome en ventana 24h. |
| ¿Qué mercados han tenido mayor volumen esta semana? | Δ volumen en 7 días. **Mejor** cuando Airflow lleva varias horas/días acumulando snapshots; con pocas horas puede salir un ranking pobre. |

Variantes que también van bien (misma lógica interna):

- `¿Qué mercados son los mas activos actualmente?`
- `top10 mercados con mas volumen en activo`
- `¿Cuáles son los 10 mercados activos con mayor volumen?`

### Volumen y liquidez

| Pregunta | Notas |
|----------|--------|
| ¿Qué mercados tuvieron más volumen en las últimas 24 horas? | Δ volumen 24h (plantilla SQL fija). |
| ¿Qué mercados tuvieron más volumen en la última semana? | Δ volumen 7 días. |
| ¿Qué mercado tuvo mayor cambio de liquidez en las últimas 24 horas? | Δ liquidez 24h. |
| ¿Qué mercado tuvo mayor cambio de liquidez en la última semana? | Δ liquidez 7 días. |
| Hola, ¿cuál es el mercado activo con más liquidez? | Mayor liquidez del último snapshot (mercado activo). |

### Noticias (HLTV, sin SQL analítico)

| Pregunta |
|----------|
| ¿Qué noticias hay de CSGO en HLTV? |
| Dame noticias recientes de Counter-Strike en HLTV |

### Conversación casual (sin base de datos)

| Pregunta |
|----------|
| Hola |
| ¿Cómo estás? |
| Gracias |

### Qué evitar en la demo (respuesta impredecible o vacía)

- Preguntas muy abiertas sin palabras clave (`mercado`, `volumen`, `probabilidad`, `activo`, `noticias`, etc.): el LLM inventa SQL.
- Mezclar en una sola frase *noticias* y *volumen/probabilidad* (el agente prioriza una u otra).
- Preguntar por mercados concretos por ID si no están en el DW cargado.

### Comprobar que hay datos antes de la demo

```bash
# Esquema (una vez)
python scripts/init_neon_schema.py

# Tras al menos una ejecución exitosa del DAG en Airflow
# (o trigger manual del DAG polymarket_csgo_hourly_pipeline)
```

En Neon deberías ver filas en `polymarket.dim_market` y `polymarket.fact_market_snapshot`. Sin snapshots, las consultas de volumen/cambios devuelven poco o nada.

En **LangGraph Studio → Chat**, si un hilo antiguo repite la misma respuesta incorrecta, pulsa **+ New Thread** y vuelve a preguntar (el estado del hilo anterior ya no contamina el turno).

## Notas importantes

- `config/` está ignorado por git (contiene contraseñas). No lo subas a GitHub.
- El sensor del pipeline usa el objeto `_SUCCESS` en S3 para garantizar que la transformación solo corre cuando el RAW está disponible.
- **Cada ejecución exitosa del DAG** (cada hora, con `catchup=False` solo la ventana programada) sigue la cadena del enunciado: extracción → RAW en S3 → sensor → **transformación y carga al Data Warehouse (Neon)**. La tarea `transform_and_load_to_neon` es la que materializa los snapshots en Postgres en cada corrida.

## Estructura

- `dags/`: DAG del pipeline
- `src/`: librería Python (cliente Polymarket, S3/Delta, transformaciones, carga a Neon)
- `sql/`: DDL del DW
- `streamlit_app/`: chatbot (LangGraph + Streamlit)
- `docs/`: arquitectura y guía de demo
