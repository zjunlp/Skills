---
name: scp-instructproteinkg
description: Use when you need to connect to the SciGraph SCP server for InstructProteinKG (UniProt/Swiss-Prot derived protein knowledge graph for sequence-text alignment and instruction learning) and call its MCP tools (query_cypher, get_kg_statistics, get_entity_details, get_experiment_workflow), including streamableHttp configuration with SCP-HUB-API-KEY and Python 3.10+ usage examples.
---

# SCP-InstructProteinKG (SciGraph) MCP client

## What this SCP is

InstructProteinKG is a protein KG for protein sequence–text alignment and instruction learning. It is extracted from UniProtKB/Swiss-Prot structured annotations and organized as `(Protein, relation, Annotation)` triples. It covers GO (BP/MF/CC) and InterPro semantics (family/superfamily/domain; conserved/active/binding sites), and introduces KCM (Knowledge Causal Modeling) to form traceable causal chains for functional/localization knowledge.

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

Example arguments (InstructProteinKG):

```json
{
  "cypher": "MATCH (e:Experiment:InstructProteinKG) RETURN e.id as experiment_id",
  "kg_name": "InstructProteinKG",
  "limit": 5
}
```

### get_kg_statistics

Return graph statistics.

Example arguments:

```json
{ "kg_name": "InstructProteinKG" }
```

### get_entity_details

Return entity details.

Example arguments:

```json
{ "entity_identifier": "experiment_1", "kg_name": "InstructProteinKG" }
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

    # Example: stats for InstructProteinKG
    result = await session.call_tool(
        "get_kg_statistics",
        arguments={"kg_name": "InstructProteinKG"},
    )
    data = json.loads(result.content[0].text)
    print(data)

    await session_ctx.__aexit__(None, None, None)
    await transport.__aexit__(None, None, None)

if __name__ == "__main__":
    asyncio.run(main())
```

## Citation

Wang, Z., Zhang, Q., Ding, K., Qin, M., Zhuang, X., Li, X., & Chen, H. (2024). InstructProtein: Aligning human and protein language via knowledge instruction. *ACL 2024 (Long Papers)*, 1114–1136. https://doi.org/10.18653/v1/2024.acl-long.62

## Reference

For the full scraped page text/schemas, read:
- `references/source.md`
