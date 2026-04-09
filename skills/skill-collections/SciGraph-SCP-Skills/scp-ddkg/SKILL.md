---
name: scp-ddkg
description: Use when you need to connect to the SciGraph SCP server for the DDKG biomedical knowledge graph (drug–drug interaction prediction) and call its MCP tools (query_cypher, get_kg_statistics, get_entity_details, get_experiment_workflow), including streamableHttp configuration with SCP-HUB-API-KEY and Python 3.10+ usage examples.
---

# SCP-DDKG (SciGraph) MCP client

## What this SCP is

DDKG (Attention-based Knowledge Graph Representation Learning for Predicting Drug-drug Interactions) is a biomedical KG for predicting drug–drug interactions (DDIs), integrating drug chemical structures (SMILES) with biomedical triples (e.g., drug–disease, drug–protein).

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
- `kg_name` (string|null, optional, default `null`): if omitted, queries across all graphs.
- `limit` (int, optional, default `100`)

Example arguments (DDKG):

```json
{
  "cypher": "MATCH (e:Experiment:DDKG) RETURN e.id as experiment_id",
  "kg_name": "DDKG",
  "limit": 5
}
```

### get_kg_statistics

Return graph statistics.

Arguments:
- `kg_name` (string|null, optional): omit for all.

Example arguments (DDKG):

```json
{ "kg_name": "DDKG" }
```

### get_entity_details

Return entity details (works across graphs).

Arguments:
- `entity_identifier` (string, required)
- `kg_name` (string|null, optional)

Example arguments (DDKG):

```json
{ "entity_identifier": "experiment_1", "kg_name": "DDKG" }
```

### get_experiment_workflow

Return the full workflow of an experiment.

Arguments:
- `experiment_id` (string, required)

Example:

```json
{ "experiment_id": "experiment_1" }
```

## Python example (streamable HTTP)

Pattern: `streamablehttp_client` + `ClientSession`.

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

    # Example: stats for DDKG
    result = await session.call_tool(
        "get_kg_statistics",
        arguments={"kg_name": "DDKG"},
    )
    data = json.loads(result.content[0].text)
    print(data)

    await session_ctx.__aexit__(None, None, None)
    await transport.__aexit__(None, None, None)

if __name__ == "__main__":
    asyncio.run(main())
```

## Citation

Su, X., Hu, L., You, Z., Hu, P., & Zhao, B. (2022). Attention-based knowledge graph representation learning for predicting drug-drug interactions. *Briefings in Bioinformatics*, 23(3), bbac140. https://doi.org/10.1093/bib/bbac140

## Reference

For the full scraped page text/schemas, read:
- `references/source.md`
