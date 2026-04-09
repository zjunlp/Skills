# Source

- URL: http://scigraph.openkg.cn/scp-instructproteinkg/
- Captured: 2026-04-09 (Asia/Shanghai)

## Page content (extracted)

### Overview

**SCP-InstructProteinKG** (Biology)

**InstructProteinKG** is a protein knowledge graph tailored for protein sequence–text alignment and instruction learning. It is primarily extracted from high-quality structured annotations of UniProtKB/Swiss-Prot, organizing associations between proteins and annotation entities as triples:

- `(Protein, relation, Annotation)`

It covers:
- The three GO branches: Biological Process / Molecular Function / Cellular Component
- InterPro semantics: family/superfamily/domain and conserved/active/binding sites

It also introduces Knowledge Causal Modeling (KCM) to convert structural features into traceable causal chains for functional/localization knowledge, enabling debiased sampling and higher-quality instruction data. Supports applications like protein functional annotation and knowledge-enhanced reasoning.

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

Demonstrates calls using `kg_name: "InstructProteinKG"`:

- `get_kg_statistics` with `{"kg_name": "InstructProteinKG"}` (or omit `kg_name` for all)
- `get_experiment_workflow` with `{"experiment_id": "experiment_1"}`
- `query_cypher` with `{"cypher": "MATCH (e:Experiment:InstructProteinKG) RETURN e.id as experiment_id", "kg_name": "InstructProteinKG", "limit": 5}`
- `get_entity_details` with `{"entity_identifier": "experiment_1", "kg_name": "InstructProteinKG"}`

#### Acknowledgements

Data contributor: Zhejiang University.

Citation (as shown on the page):
- Wang, Z., Zhang, Q., Ding, K., Qin, M., Zhuang, X., Li, X., & Chen, H. (2024). InstructProtein: Aligning human and protein language via knowledge instruction. In L.-W. Ku, A. Martins, & V. Srikumar (Eds.), *Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)* (pp. 1114–1136). Association for Computational Linguistics. https://doi.org/10.18653/v1/2024.acl-long.62

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
