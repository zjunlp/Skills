---
name: scp-kg-fm
description: Use when you need to connect to the SciGraph SCP server for KG-FM (framework-material knowledge graph dataset; see npj Computational Materials 2025 paper) and call its MCP tools (query_cypher, get_kg_statistics, get_entity_details, get_experiment_workflow), including streamableHttp configuration with SCP-HUB-API-KEY and Python 3.10+ usage examples.
---

# SCP-KG-FM (SciGraph) MCP client

## What this SCP is

KG-FM is described as a framework-material knowledge graph dataset. The page cites a work on constructing a knowledge graph for framework materials enabled by large language models.

Note: the page’s short description text appears inconsistent (it looks like reused YaSAScore copy), but the tool usage and citation blocks clearly reference KG-FM.

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

Example arguments (KG-FM):

```json
{
  "cypher": "MATCH (e:Experiment:KG-FM) RETURN e.id as experiment_id",
  "kg_name": "KG-FM",
  "limit": 5
}
```

### get_kg_statistics

Return graph statistics.

Example arguments:

```json
{ "kg_name": "KG-FM" }
```

### get_entity_details

Return entity details.

Example arguments:

```json
{ "entity_identifier": "experiment_1", "kg_name": "KG-FM" }
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

    # Example: stats for KG-FM
    result = await session.call_tool(
        "get_kg_statistics",
        arguments={"kg_name": "KG-FM"},
    )
    data = json.loads(result.content[0].text)
    print(data)

    await session_ctx.__aexit__(None, None, None)
    await transport.__aexit__(None, None, None)

if __name__ == "__main__":
    asyncio.run(main())
```

## Citation

Bai, X., He, S., Li, Y. et al. (2025). Construction of a knowledge graph for framework material enabled by large language models and its application. *npj Computational Materials*, 11, 51. https://doi.org/10.1038/s41524-025-01540-6

## Reference

For the full scraped page text, read:
- `references/source.md`
