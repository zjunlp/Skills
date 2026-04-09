---
name: scp-proteinkg25
description: Use when you need to connect to the SciGraph SCP server for ProteinKG25 (GO + protein sequence knowledge graph) and call its MCP tools (query_cypher, get_kg_statistics, get_entity_details, get_experiment_workflow), including streamableHttp configuration with SCP-HUB-API-KEY and Python 3.10+ usage examples.
---

# SCP-ProteinKG25 (SciGraph) MCP client

## What this SCP is

ProteinKG25 is a large-scale KG integrating Gene Ontology (GO) structure with protein sequences and textual GO definitions. It aligns GO terms and proteins via internal GO–GO relations (e.g., `is_a`, `part_of`) and external Protein–GO annotations. It’s used for knowledge-enhanced pretraining of protein language models (e.g., OntoProtein).

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

Example arguments (ProteinKG25):

```json
{
  "cypher": "MATCH (e:Experiment:ProteinKG25) RETURN e.id as experiment_id",
  "kg_name": "ProteinKG25",
  "limit": 5
}
```

### get_kg_statistics

Return graph statistics.

Example arguments (ProteinKG25):

```json
{ "kg_name": "ProteinKG25" }
```

### get_entity_details

Return entity details.

Example arguments:

```json
{ "entity_identifier": "experiment_1", "kg_name": "ProteinKG25" }
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

    # Example: stats for ProteinKG25
    result = await session.call_tool(
        "get_kg_statistics",
        arguments={"kg_name": "ProteinKG25"},
    )
    data = json.loads(result.content[0].text)
    print(data)

    await session_ctx.__aexit__(None, None, None)
    await transport.__aexit__(None, None, None)

if __name__ == "__main__":
    asyncio.run(main())
```

## Citation

Zhang, N., Bi, Z., Liang, X., Cheng, S., Hong, H., Deng, S., Zhang, Q., Lian, J., & Chen, H. (2022). *OntoProtein: Protein pretraining with gene ontology embedding*. ICLR. https://openreview.net/forum?id=yfe1VMYAXa4

## Reference

For the full scraped page text/schemas, read:
- `references/source.md`
