---
name: scp-ikraph
description: Use when you need to connect to the SciGraph SCP server for iKraph (large-scale biomedical knowledge graph from PubMed + external databases like Hetionet) and call its MCP tools (query_cypher, get_kg_statistics, get_entity_details, get_experiment_workflow), including streamableHttp configuration with SCP-HUB-API-KEY and Python 3.10+ usage examples.
---

# SCP-iKraph (SciGraph) MCP client

## What this SCP is

iKraph is a comprehensive, large-scale biomedical knowledge graph for AI-powered and data-driven biomedical research. It integrates structured relationships extracted from PubMed abstracts and external biomedical databases (e.g., Hetionet), covering entities such as genes, drugs, chemicals, and diseases.

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

Example arguments (iKraph):

```json
{
  "cypher": "MATCH (e:Experiment:iKraph) RETURN e.id as experiment_id",
  "kg_name": "iKraph",
  "limit": 5
}
```

### get_kg_statistics

Return graph statistics.

Example arguments:

```json
{ "kg_name": "iKraph" }
```

### get_entity_details

Return entity details.

Example arguments:

```json
{ "entity_identifier": "experiment_1", "kg_name": "iKraph" }
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

    # Example: stats for iKraph
    result = await session.call_tool(
        "get_kg_statistics",
        arguments={"kg_name": "iKraph"},
    )
    data = json.loads(result.content[0].text)
    print(data)

    await session_ctx.__aexit__(None, None, None)
    await transport.__aexit__(None, None, None)

if __name__ == "__main__":
    asyncio.run(main())
```

## Citation

Zhang, Y., Sui, X., Pan, F. et al. A comprehensive large-scale biomedical knowledge graph for AI-powered data-driven biomedical research. *Nat Mach Intell* 7, 602–614 (2025). https://doi.org/10.1038/s42256-025-01014-w

## Reference

For the full scraped page text/schemas, read:
- `references/source.md`
