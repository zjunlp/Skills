---
name: scp-mekg
description: Use when you need to connect to the SciGraph SCP server for MEKG (Materials Experiment Knowledge Graph capturing provenance of synthesis/processing/characterization/performance experiments) and call its MCP tools (query_cypher, get_kg_statistics, get_entity_details, get_experiment_workflow), including streamableHttp configuration with SCP-HUB-API-KEY and Python 3.10+ usage examples.
---

# SCP-MEKG (SciGraph) MCP client

## What this SCP is

MEKG (Materials Experiment Knowledge Graph) encodes complete provenance for materials science experiments, including synthesis, processing, characterization, and performance analysis.

It models hierarchical relationships among material samples, experimental processes, and derived data to support querying and knowledge discovery in high-throughput experimental materials science.

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

Example arguments (MEKG):

```json
{
  "cypher": "MATCH (e:Experiment:MEKG) RETURN e.id as experiment_id",
  "kg_name": "MEKG",
  "limit": 5
}
```

### get_kg_statistics

Return graph statistics.

Example arguments:

```json
{ "kg_name": "MEKG" }
```

### get_entity_details

Return entity details.

Example arguments:

```json
{ "entity_identifier": "experiment_1", "kg_name": "MEKG" }
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

    # Example: stats for MEKG
    result = await session.call_tool(
        "get_kg_statistics",
        arguments={"kg_name": "MEKG"},
    )
    data = json.loads(result.content[0].text)
    print(data)

    await session_ctx.__aexit__(None, None, None)
    await transport.__aexit__(None, None, None)

if __name__ == "__main__":
    asyncio.run(main())
```

## Citation

Statt, M. J., Rohr, B. A., Guevarra, D., Breeden, J., Suram, S. K., & Gregoire, J. M. (2023). The materials experiment knowledge graph. *Digital Discovery*, 2(4), 909–914. https://doi.org/10.1039/D3DD00067B

## Reference

For the full scraped page text, read:
- `references/source.md`
