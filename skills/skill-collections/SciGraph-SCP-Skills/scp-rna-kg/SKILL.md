---
name: scp-rna-kg
description: Use when you need to connect to the SciGraph SCP server for RNA-KG (RNA-centered heterogeneous biomedical knowledge graph with contextual properties for RNA interaction contexts) and call its MCP tools (query_cypher, get_kg_statistics, get_entity_details, get_experiment_workflow), including streamableHttp configuration with SCP-HUB-API-KEY and Python 3.10+ usage examples.
---

# SCP-RNA-KG (SciGraph) MCP client

## What this SCP is

RNA-KG is an RNA-centered heterogeneous biomedical knowledge graph. It combines standard entity–relationship modeling with contextual properties to capture the specific biological contexts in which RNA interactions occur, addressing the common “disconnection from context” issue in biomedical datasets.

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

Example arguments (RNA-KG):

```json
{
  "cypher": "MATCH (e:Experiment:RNA-KG) RETURN e.id as experiment_id",
  "kg_name": "RNA-KG",
  "limit": 5
}
```

### get_kg_statistics

Return graph statistics.

Example arguments:

```json
{ "kg_name": "RNA-KG" }
```

### get_entity_details

Return entity details.

Example arguments:

```json
{ "entity_identifier": "experiment_1", "kg_name": "RNA-KG" }
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

    # Example: stats for RNA-KG
    result = await session.call_tool(
        "get_kg_statistics",
        arguments={"kg_name": "RNA-KG"},
    )
    data = json.loads(result.content[0].text)
    print(data)

    await session_ctx.__aexit__(None, None, None)
    await transport.__aexit__(None, None, None)

if __name__ == "__main__":
    asyncio.run(main())
```

## Citation

Cavalleri, E., Cabri, A., Soto-Gomez, M. et al. An ontology-based knowledge graph for representing interactions involving RNA molecules. *Sci Data* 11, 906 (2024). https://doi.org/10.1038/s41597-024-03673-7

## Reference

For the full scraped page text/schemas, read:
- `references/source.md`
