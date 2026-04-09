---
name: scp-otter-primekg
description: Use when you need to connect to the SciGraph SCP server for Otter-PrimeKG (IBM Research Otter-Knowledge multimodal knowledge graphs for drug discovery) and call its MCP tools (query_cypher, get_kg_statistics, get_entity_details, get_experiment_workflow), including streamableHttp configuration with SCP-HUB-API-KEY and Python 3.10+ usage examples.
---

# SCP-Otter-PrimeKG (SciGraph) MCP client

## What this SCP is

Otter-Knowledge is a collection of multimodal knowledge graphs (MKGs) from IBM Research for drug discovery. It integrates structured biomedical knowledge to improve representation learning for protein sequences and drug SMILES. The page notes four graphs (Otter UBC, Otter PrimeKG, Otter DUDe, Otter STITCH), over 30M triples total, with nodes annotated by modal type (sequence/SMILES/text/numerical attributes, etc.), and SOTA results on TDC drug-target binding affinity benchmarks.

## Connection info

- MCP server URL:
  - `https://scp.intern-ai.org.cn/api/v1/mcp/37/SciGraph`
- Auth header:
  - `SCP-HUB-API-KEY: {API-KEY}`

## Install

```bash
pip install mcp
```

## Configure (MCP config JSON)

```json
{
  "mcpServers": {
    "SciGraph": {
      "type": "streamableHttp",
      "description": "这是一款面向科学研究的统一知识查询服务，集成了化学、生物等多个学科领域的知识图谱数据，支持跨学科知识检索、实体关系查询、领域知识问答等操作",
      "url": "https://scp.intern-ai.org.cn/api/v1/mcp/37/SciGraph",
      "headers": {
        "SCP-HUB-API-KEY": "{API-KEY}"
      }
    }
  }
}
```

## Tools

### query_cypher

Execute a Cypher query and return JSON results.

Arguments:
- `cypher` (string, required)
- `kg_name` (string|null, optional, default `null`)
- `limit` (int, optional, default `100`)

Example arguments (Otter-PrimeKG):

```json
{
  "cypher": "MATCH (e:Experiment:Otter-PrimeKG) RETURN e.id as experiment_id",
  "kg_name": "Otter-PrimeKG",
  "limit": 5
}
```

### get_kg_statistics

Return graph statistics.

Example arguments:

```json
{ "kg_name": "Otter-PrimeKG" }
```

### get_entity_details

Return entity details.

Example arguments:

```json
{ "entity_identifier": "experiment_1", "kg_name": "Otter-PrimeKG" }
```

### get_experiment_workflow

Return the full workflow of an experiment.

Example arguments:

```json
{ "experiment_id": "experiment_1" }
```

## Python example (streamable HTTP)

```python
import asyncio
import json
from mcp.client.streamable_http import streamablehttp_client
from mcp.client.session import ClientSession

SERVER_URL = "https://scp.intern-ai.org.cn/api/v1/mcp/37/SciGraph"

async def main():
    transport = streamablehttp_client(
        url=SERVER_URL,
        headers={"SCP-HUB-API-KEY": "sk-xxx"},
    )
    read, write, get_session_id = await transport.__aenter__()

    session_ctx = ClientSession(read, write)
    session = await session_ctx.__aenter__()
    await session.initialize()

    # Example: stats for Otter-PrimeKG
    result = await session.call_tool(
        "get_kg_statistics",
        arguments={"kg_name": "Otter-PrimeKG"},
    )
    data = json.loads(result.content[0].text)
    print(data)

    await session_ctx.__aexit__(None, None, None)
    await transport.__aexit__(None, None, None)

if __name__ == "__main__":
    asyncio.run(main())
```

## Citation

Hoang, T. L., Sbodio, M. L., Martinez Galindo, M., Zayats, M., Fernandez-Diaz, R., Valls, V., Picco, G., Berrospi, C., & Lopez, V. (2024). Knowledge Enhanced Representation Learning for Drug Discovery. *Proceedings of the AAAI Conference on Artificial Intelligence*, 38(9), 10544–10552. https://doi.org/10.1609/aaai.v38i9.28924

## Reference

For the full scraped page text/schemas, read:
- `references/source.md`
