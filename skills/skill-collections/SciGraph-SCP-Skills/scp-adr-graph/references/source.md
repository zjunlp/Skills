# SCP-ADR-Graph source (scraped)

Source URL: http://scigraph.openkg.cn/scp-adr-graph/

> NOTE: This is scraped from an external webpage and may include formatting artifacts.

## Page content (readable extract)

🧪 A knowledge graph dataset focusing on the relationship between drugs and adverse reactions, used in conjunction with the EdgePrediction Python library. This dataset aims to support link prediction tasks based on statistical enrichment analysis, primarily for discovering potential unknown side effects of drugs.

### Tool list

- query_cypher — Execute any Cypher query statement, supporting flexible graph database operations
- get_kg_statistics — Obtain statistical information such as nodes, relationships, and type distribution of the knowledge graph
- get_entity_details — Obtain detailed information and relationships of entities based on entity identifiers
- get_experiment_workflow — Obtain the complete workflow of chemical experiments

### Usage (ADR-Graph examples)

(See browser snapshot for full code blocks.)

### Acknowledgement & Reference

The contributors of this knowledge graph data is King’s College London. Please cite the original paper when using this data:

Bean, D.M., Wu, H., Iqbal, E. et al. (2017). Knowledge graph prediction of unknown adverse drug reactions and validation in electronic health records. *Scientific Reports*, 7, 16416. https://doi.org/10.1038/s41598-017-16674-x
