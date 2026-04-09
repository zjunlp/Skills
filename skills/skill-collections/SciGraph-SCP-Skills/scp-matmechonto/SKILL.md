---
name: scp-matmechonto
description: Use when you need to connect to the SciGraph SCP server for MatMechOnto (MaterioMiner dataset linking a materials mechanics ontology with fine-grained literature entity annotations; supports NER and CPMP relation extraction) and call its MCP tools (query_cypher, get_kg_statistics, get_entity_details, get_experiment_workflow), including streamableHttp configuration with SCP-HUB-API-KEY and Python 3.10+ usage examples.
---

# SCP-MatMechOnto (SciGraph) MCP client

## What this SCP is

MatMechOnto (MaterioMiner) is a dataset that links a custom materials mechanics ontology with entities annotated from scientific literature.

The page describes extremely fine-grained annotation (179 classes) by three domain experts across four publications, resulting in 2,191 curated entities. It supports tasks such as:

- Fine-grained named entity recognition
- Causal CPMP (composition-process-microstructure-property) relationship extraction
- Neurosymbolic AI in materials science

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

Example arguments (MatMechOnto):

```json
{
  "cypher": "MATCH (e:Experiment:MatMechOnto) RETURN e.id as experiment_id",
  "kg_name": "MatMechOnto",
  "limit": 5
}
```

### get_kg_statistics

Return graph statistics.

Example arguments:

```json
{ "kg_name": "MatMechOnto" }
```

### get_entity_details

Return entity details.

Example arguments:

```json
{ "entity_identifier": "experiment_1", "kg_name": "MatMechOnto" }
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

    # Example: stats for MatMechOnto
    result = await session.call_tool(
        "get_kg_statistics",
        arguments={"kg_name": "MatMechOnto"},
    )
    data = json.loads(result.content[0].text)
    print(data)

    await session_ctx.__aexit__(None, None, None)
    await transport.__aexit__(None, None, None)

if __name__ == "__main__":
    asyncio.run(main())
```

## Citation

Durmaz, A.R., Thomas, A., Mishra, L. et al. (2024). An ontology-based text mining dataset for extraction of process-structure-property entities. *Scientific Data*, 11, 1112. https://doi.org/10.1038/s41597-024-03926-5

## Reference

For the full scraped page text, read:
- `references/source.md`
