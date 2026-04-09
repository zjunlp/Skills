---
name: scp-pertkge
description: Use when you need to connect to the SciGraph SCP server for PertKGE (knowledge graph for compound-protein interaction inference using perturbation transcriptomics + regulatory network; cold-start CPI prediction) and call its MCP tools (query_cypher, get_kg_statistics, get_entity_details, get_experiment_workflow), including streamableHttp configuration with SCP-HUB-API-KEY and Python 3.10+ usage examples.
---

# SCP-PertKGE (SciGraph) MCP client

## What this SCP is

PertKGE is a knowledge graph for inferring compound-protein interactions (CPI). It integrates perturbed transcriptomics with a refined biological regulatory network (DNA, mRNA, TF, RBP, etc.) to simulate cellular response processes, targeting “cold start” CPI prediction for new drugs or new targets.

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

Example arguments (PertKGE):

```json
{
  "cypher": "MATCH (e:Experiment:PertKGE) RETURN e.id as experiment_id",
  "kg_name": "PertKGE",
  "limit": 5
}
```

### get_kg_statistics

Return graph statistics.

Example arguments:

```json
{ "kg_name": "PertKGE" }
```

### get_entity_details

Return entity details.

Example arguments:

```json
{ "entity_identifier": "experiment_1", "kg_name": "PertKGE" }
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

    # Example: stats for PertKGE
    result = await session.call_tool(
        "get_kg_statistics",
        arguments={"kg_name": "PertKGE"},
    )
    data = json.loads(result.content[0].text)
    print(data)

    await session_ctx.__aexit__(None, None, None)
    await transport.__aexit__(None, None, None)

if __name__ == "__main__":
    asyncio.run(main())
```

## Citation

Ni, S., Kong, X., Zhang, Y., Chen, Z., Wang, Z., Fu, Z., Huo, R., Tong, X., Qu, N., Wu, X., Wang, K., Zhang, W., Zhang, R., Zhang, Z., Shi, J., Wang, Y., Yang, R., Li, X., Zhang, S., & Zheng, M. (2024). Identifying compound-protein interactions with knowledge graph embedding of perturbation transcriptomics. *Cell Genomics*, 4(10), 100655. https://doi.org/10.1016/j.xgen.2024.100655

## Reference

For the full scraped page text/schemas, read:
- `references/source.md`
