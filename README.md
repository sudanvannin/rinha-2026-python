# Rinha de Backend 2026 - baseline

API em Python para o desafio de fraude da Rinha de Backend 2026.

Arquitetura local:

- `nginx` na porta `9999`
- `api1`, `api2` e `api3` em Python
- servidor HTTP proprio com `asyncio`
- `orjson` para parse de JSON
- `scipy.spatial.cKDTree` para o KNN em um indice compacto de candidatos de borda gerado a partir de `resources/references.json.gz`
- fallback por score fora da zona ambigua para reduzir fila no p99
- vetores de entrada arredondados em 4 casas para reproduzir a rotulagem oficial

## Rodar

```bash
docker compose up --build
```

Endpoints:

- `GET /ready`
- `POST /fraud-score`

## Regerar o indice

```bash
python scripts/build_candidates.py \
  --references _official/resources/references.json.gz \
  --output data/candidates.bin
```

O indice nao usa payloads de teste; ele e derivado apenas das referencias oficiais.

## Testes locais

```bash
python -m unittest discover -s tests
```
