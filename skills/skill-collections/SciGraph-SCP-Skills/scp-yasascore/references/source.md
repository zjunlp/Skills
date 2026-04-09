# SCP-YaSAScore source (scraped)

Source URL: http://scigraph.openkg.cn/scp-yasascore/

> NOTE: This is scraped from an external webpage and may include formatting artifacts.

## Page content (readable extract)

🧪 YaSAScore Reaction Knowledge Graph is a domain-specific knowledge graph built from chemical reaction data to support prediction of compound synthesis accessibility (SA). It integrates reactions from the USPTO and Pistachio datasets into a directed molecular network (compounds as nodes; reactions as directed edges from reactants to products).

It enables shortest-path-based estimation of synthetic complexity and is used for training/benchmarking ML models (e.g., CMPNN and SYBA-2).

### Tool list

- query_cypher — Execute any Cypher query statement, supporting flexible graph database operations
- get_kg_statistics — Obtain statistical information such as nodes, relationships, and type distribution of the knowledge graph
- get_entity_details — Obtain detailed information and relationships of entities based on entity identifiers
- get_experiment_workflow — Obtain the complete workflow of chemical experiments

### Usage (YaSAScore examples)

(See browser snapshot for full code blocks.)

### Acknowledgement & Reference

The contributors of this knowledge graph data is Guangdong Provincial Key Laboratory of Laboratory Animals. Please cite the original paper when using this data:

Li, B., & Chen, H. (2022). Prediction of compound synthesis accessibility based on reaction knowledge graph. *Molecules*, 27(3), 1039. https://doi.org/10.3390/molecules27031039
