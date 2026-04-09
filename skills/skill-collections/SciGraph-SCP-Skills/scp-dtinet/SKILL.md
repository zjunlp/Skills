---
name: scp-dtinet
description: Use when you need to connect to the SciGraph SCP server for DTINet (Drug-Target Interaction Network knowledge graph for predicting novel drug–target interactions from heterogeneous biomedical networks) and call its MCP tools (query_cypher, get_kg_statistics, get_entity_details, get_experiment_workflow), including streamableHttp configuration with SCP-HUB-API-KEY and Python 3.10+ usage examples.
---

# SCP-DTINet (SciGraph) MCP client

## What this SCP is

DTINet (Drug-Target Interaction Network) is a computational framework/dataset for predicting novel drug–target interactions (DTIs). It integrates multi-source information from heterogeneous biological networks, learns low-dimensional feature vectors for drug/protein nodes, and predicts new DTIs via a vector space projection scheme. It is used for drug repositioning and computational drug discovery.

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

Example arguments (DTINet):

```json
{
  "cypher": "MATCH (e:Experiment:DTINet) RETURN e.id as experiment_id",
  "kg_name": "DTINet",
  "limit": 5
}
```

### get_kg_statistics

Return graph statistics.

Example arguments:

```json
{ "kg_name": "DTINet" }
```

### get_entity_details

Return entity details.

Example arguments:

```json
{ "entity_identifier": "experiment_1", "kg_name": "DTINet" }
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

    # Example: stats for DTINet
    result = await session.call_tool(
        "get_kg_statistics",
        arguments={"kg_name": "DTINet"},
    )
    data = json.loads(result.content[0].text)
    print(data)

    await session_ctx.__aexit__(None, None, None)
    await transport.__aexit__(None, None, None)

if __name__ == "__main__":
    asyncio.run(main())
```

## Citation

Luo, Y., Zhao, X., Zhou, J. et al. (2017). A network integration approach for drug-target interaction prediction and computational drug repositioning from heterogeneous information. *Nature Communications*, 8, 573. https://doi.org/10.1038/s41467-017-00680-8

## Reference

For the full scraped page text, read:
- `references/source.md`
