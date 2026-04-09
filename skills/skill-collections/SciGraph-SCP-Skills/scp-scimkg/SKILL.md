---
name: scp-scimkg
description: Use when you need to connect to the SciGraph SCP server for SciMKG (multimodal K-12 science education knowledge graph; packaged as SciMKG.ttl RDF/Turtle view) and call its MCP tools (query_cypher, get_kg_statistics, get_entity_details, get_experiment_workflow), including streamableHttp configuration with SCP-HUB-API-KEY and Python 3.10+ usage examples.
---

# SCP-SciMKG (SciGraph) MCP client

## What this SCP is

SciMKG is a large-scale multimodal educational knowledge graph for K-12 science education (text, image, video, audio). The page notes the packaged version is based on the `SciMKG.ttl` RDF/Turtle graph view (not the full raw multimodal source package).

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

Example arguments (SciMKG):

```json
{
  "cypher": "MATCH (e:Experiment:SciMKG) RETURN e.id as experiment_id",
  "kg_name": "SciMKG",
  "limit": 5
}
```

### get_kg_statistics

Return graph statistics.

Example arguments:

```json
{ "kg_name": "SciMKG" }
```

### get_entity_details

Return entity details.

Example arguments:

```json
{ "entity_identifier": "experiment_1", "kg_name": "SciMKG" }
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

    # Example: stats for SciMKG
    result = await session.call_tool(
        "get_kg_statistics",
        arguments={"kg_name": "SciMKG"},
    )
    data = json.loads(result.content[0].text)
    print(data)

    await session_ctx.__aexit__(None, None, None)
    await transport.__aexit__(None, None, None)

if __name__ == "__main__":
    asyncio.run(main())
```

## Citation

Lu, T., Wang, Z., Zhou, Y., Guan, Y., Bai, Z., & Du, J. (2026). SciMKG: A Multimodal Knowledge Graph for Science Education with Text, Image, Video and Audio. *Proceedings of the AAAI Conference on Artificial Intelligence*, 40(18), 15466–15474. https://doi.org/10.1609/aaai.v40i18.38574

## Reference

For the full scraped page text, read:
- `references/source.md`
