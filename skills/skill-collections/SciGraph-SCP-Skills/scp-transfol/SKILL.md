---
name: scp-transfol
description: Use when you need to connect to the SciGraph SCP server for TransFOL (multimodal biomedical knowledge graph for drug-drug interaction prediction and complex relational/logical reasoning) and call its MCP tools (query_cypher, get_kg_statistics, get_entity_details, get_experiment_workflow), including streamableHttp configuration with SCP-HUB-API-KEY and Python 3.10+ usage examples.
---

# SCP-TransFOL (SciGraph) MCP client

## What this SCP is

TransFOL is a comprehensive multimodal biomedical knowledge graph designed for drug-drug interaction (DDI) prediction and complex relational reasoning. It integrates structured knowledge from DrugBank, TWOSIDES, DrugCombDB, and Phenomebrowser, covering drugs, proteins, genes, diseases, phenotypes, enzymes, and biological pathways.

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

Example arguments (TransFOL):

```json
{
  "cypher": "MATCH (e:Experiment:TransFOL) RETURN e.id as experiment_id",
  "kg_name": "TransFOL",
  "limit": 5
}
```

### get_kg_statistics

Return graph statistics.

Example arguments:

```json
{ "kg_name": "TransFOL" }
```

### get_entity_details

Return entity details.

Example arguments:

```json
{ "entity_identifier": "experiment_1", "kg_name": "TransFOL" }
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

    # Example: stats for TransFOL
    result = await session.call_tool(
        "get_kg_statistics",
        arguments={"kg_name": "TransFOL"},
    )
    data = json.loads(result.content[0].text)
    print(data)

    await session_ctx.__aexit__(None, None, None)
    await transport.__aexit__(None, None, None)

if __name__ == "__main__":
    asyncio.run(main())
```

## Citation

Cheng, J., Zhang, Y., Zhang, H., Ji, S., & Lu, M. (2024). TransFOL: A logical query model for complex relational reasoning in drug-drug interaction. *IEEE Journal of Biomedical and Health Informatics*, 28(8), 49754985. https://doi.org/10.1109/JBHI.2024.3401035

## Reference

For the full scraped page text, read:
- `references/source.md`
