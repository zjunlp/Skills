# Source

- URL: http://scigraph.openkg.cn/scp-clinicalkg/
- Captured: 2026-04-09 (Asia/Shanghai)

## Page content (extracted)

### Overview

**SCP-ClinicalKG** (Biology)

The page describes a complete database dump of the **Clinical Knowledge Graph (CKG)** for quickly constructing a Neo4j graph database.

CKG is an open-source platform integrating:
- 9 ontologies
- 26 relevant biomedical databases
- experimental data

Goal: accelerate proteomics workflow analysis and clinical decision-making via graph representation and machine learning algorithms.

#### Tool list (as shown on the page)

- **query_cypher**: Execute any Cypher query statement.
- **get_kg_statistics**: Get statistics (nodes/relationships/type distribution).
- **get_entity_details**: Get entity details + relationships by identifier.
- **get_experiment_workflow**: Get complete experiment workflow.

#### Quick start

1) **Dependence**

Recommended: Python 3.10+

Install:

```bash
pip install mcp
```

2) **Configuration**

Example Python client uses:
- `mcp.client.streamable_http.streamablehttp_client`
- `mcp.client.session.ClientSession`

Default server URL in example:

```
https://scp.intern-ai.org.cn/api/v1/mcp/37/SciGraph
```

Auth header in example:

```
SCP-HUB-API-KEY: sk-xxx
```

3) **Usage examples (Python)**

Demonstrates calls using `kg_name: "ClinicalKG"`:

- `get_kg_statistics` with `{"kg_name": "ClinicalKG"}` (or omit `kg_name` for all)
- `get_experiment_workflow` with `{"experiment_id": "experiment_1"}`
- `query_cypher` with `{"cypher": "MATCH (e:Experiment:ClinicalKG) RETURN e.id as experiment_id", "kg_name": "ClinicalKG", "limit": 5}`
- `get_entity_details` with `{"entity_identifier": "experiment_1", "kg_name": "ClinicalKG"}`

#### Aknowledgement & Reference

Contributor: University of Copenhagen.

Citation:
- Santos, A., Colaço, A. R., Nielsen, A. B., Niu, L., Geyer, P. E., Coscia, F., Albrechtsen, N. J. W., Mundt, F., Jensen, L. J., & Mann, M. (2020). Clinical knowledge graph integrates proteomics data into clinical decision-making. *bioRxiv*. https://doi.org/10.1101/2020.05.09.084897

#### “How to use?”

1) Install MCP SDK: `pip install mcp` (links to SDK docs)

2) Apply for an API key (links to application portal)

3) Configuration information:

- URL: `https://scp.intern-ai.org.cn/api/v1/mcp/37/SciGraph`

Example MCP config:

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

### Tool tab

#### query_cypher

"Execute Cypher query and return results."

Arguments:
- `cypher` (string, required): Cypher query statement.
- `kg_name` (string|null, optional, default null): Knowledge graph name; if omitted, queries on all graphs. Supported (as shown on page): `ElementKG`, `InstructProteinKG`.
- `limit` (integer, optional, default 100): Max results.

Returns: JSON query results.

Schema:

```json
{ "properties": { "cypher": { "type": "string" }, "kg_name": { "anyOf": [ { "type": "string" }, { "type": "null" } ], "default": null }, "limit": { "default": 100, "type": "integer" } }, "required": [ "cypher" ], "type": "object" }
```

#### get_kg_statistics

"Obtain statistical information of the knowledge graph."

Arguments:
- `kg_name` (string|null, optional, default null): if omitted, returns all graphs. Supported (as shown on page): `ElementKG`, `InstructProteinKG`.

Returns: JSON statistics.

Schema:

```json
{ "properties": { "kg_name": { "anyOf": [ { "type": "string" }, { "type": "null" } ], "default": null } }, "type": "object" }
```

#### get_entity_details

"Retrieve detailed information of an entity (support all knowledge graphs)."

Args:
- `entity_identifier` (string, required)
  - ElementKG: entity id (e.g. `experiment_1`)
  - InstructProteinKG: protein sequence or hash
- `kg_name` (string|null, optional, default null): if omitted, search across all graphs. Supports (as shown on page): `ElementKG`, `InstructProteinKG`.

Returns: JSON details.

Schema:

```json
{ "properties": { "entity_identifier": { "type": "string" }, "kg_name": { "anyOf": [ { "type": "string" }, { "type": "null" } ], "default": null } }, "required": [ "entity_identifier" ], "type": "object" }
```

#### get_experiment_workflow

"Get the complete workflow of the experiment (exclusive for ElementKG)."

Args:
- `experiment_id` (string, required)

Returns: JSON workflow (steps + reagents)

Schema:

```json
{ "properties": { "experiment_id": { "type": "string" } }, "required": [ "experiment_id" ], "type": "object" }
```
