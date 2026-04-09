---
name: scp-mkg-fenn
description: Use when you need to connect to the SciGraph SCP server for MKG-FENN (multimodal knowledge graph fused end-to-end neural network for drug-drug interaction prediction) and call its MCP tools (query_cypher, get_kg_statistics, get_entity_details, get_experiment_workflow), including streamableHttp configuration with SCP-HUB-API-KEY and Python 3.10+ usage examples.
---

# SCP-MKG-FENN (SciGraph) MCP client

## What this SCP is

MKG-FENN is a multimodal knowledge graph fusion system for accurate drug-drug interaction (DDI) prediction. It integrates structured and semi-structured knowledge from multiple data sources, including drug information, protein targets, biological pathways, molecular properties, and clinical interaction data.

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

Example arguments (MKG-FENN):

```json
{
  "cypher": "MATCH (e:Experiment:MKG-FENN) RETURN e.id as experiment_id",
  "kg_name": "MKG-FENN",
  "limit": 5
}
```

### get_kg_statistics

Return graph statistics.

Example arguments:

```json
{ "kg_name": "MKG-FENN" }
```

### get_entity_details

Return entity details.

Example arguments:

```json
{ "entity_identifier": "experiment_1", "kg_name": "MKG-FENN" }
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

    # Example: stats for MKG-FENN
    result = await session.call_tool(
        "get_kg_statistics",
        arguments={"kg_name": "MKG-FENN"},
    )
    data = json.loads(result.content[0].text)
    print(data)

    await session_ctx.__aexit__(None, None, None)
    await transport.__aexit__(None, None, None)

if __name__ == "__main__":
    asyncio.run(main())
```

## Citation

Wu, D., Sun, W., He, Y., Chen, Z., & Luo, X. (2024). MKG-FENN: A Multimodal Knowledge Graph Fused End-to-End Neural Network for Accurate Drug–Drug Interaction Prediction. *Proceedings of the AAAI Conference on Artificial Intelligence*, 38(9), 10216–10224. https://doi.org/10.1609/aaai.v38i9.28887

## Reference

For the full scraped page text/schemas, read:
- `references/source.md`
