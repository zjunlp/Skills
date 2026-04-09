---
name: scp-elementkg-2
description: Use when you need to connect to the SciGraph SCP server for ElementKG 2.0 / InstructProteinKG and call its MCP tools (query_cypher, get_kg_statistics, get_entity_details, get_experiment_workflow), including configuration (streamableHttp + SCP-HUB-API-KEY) and Python 3.10+ client usage examples.
---

# SCP-ElementKG 2.0 (SciGraph) MCP client

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

## Available tools

### query_cypher

Execute a Cypher query and return JSON results.

Arguments:
- `cypher` (string, required)
- `kg_name` (string|null, optional, default `null`): if omitted, queries across all graphs. Supported: `ElementKG`, `InstructProteinKG`.
- `limit` (int, optional, default `100`)

Example arguments:

```json
{
  "cypher": "MATCH (e:Experiment:ElementKG) RETURN e.id as experiment_id",
  "kg_name": "ElementKG",
  "limit": 5
}
```

### get_kg_statistics

Return graph statistics.

Arguments:
- `kg_name` (string|null, optional): omit for all. Supported: `ElementKG`, `InstructProteinKG`.

### get_entity_details

Return entity details (works across graphs).

Arguments:
- `entity_identifier` (string, required)
  - ElementKG: entity id (e.g. `experiment_1`)
  - InstructProteinKG: protein sequence or hash
- `kg_name` (string|null, optional): omit to search across all graphs.

### get_experiment_workflow

Return the full workflow of an ElementKG experiment.

Arguments:
- `experiment_id` (string, required)

## Python example (streamable HTTP)

Use `mcp.client.streamable_http.streamablehttp_client` + `mcp.client.session.ClientSession`.

Minimal pattern:

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

    # Example: get stats
    result = await session.call_tool(
        "get_kg_statistics",
        arguments={"kg_name": "ElementKG"},
    )

    # Parse JSON payload
    data = json.loads(result.content[0].text)
    print(data)

    await session_ctx.__aexit__(None, None, None)
    await transport.__aexit__(None, None, None)

if __name__ == "__main__":
    asyncio.run(main())
```

## Reference

If you need the exact scraped wording / schemas from the page, read:
- `references/source.md`
