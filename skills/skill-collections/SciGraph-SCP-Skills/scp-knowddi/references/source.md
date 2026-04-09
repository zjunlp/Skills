# SCP-KnowDDI (SciGraph) — Page Content (captured 2026-04-09)

## Summary
KnowDDI is a knowledge graph-based system for accurate and interpretable drug-drug interaction (DDI) prediction using knowledge subgraph learning. It integrates biomedical knowledge from Hetionet, DrugBank, and BioSNAP to identify potential drug interactions and support zero-shot DDI prediction.

## KnowDDI SCP Server
KnowDDI is a knowledge graph-based system for accurate and interpretable drug-drug interaction (DDI) prediction using knowledge subgraph learning. It integrates biomedical knowledge from Hetionet, DrugBank, and BioSNAP to identify potential drug interactions and support zero-shot DDI prediction.

## Tool List
| Tool Name | Functional Description |
|---|---|
| `query_cypher` | Execute any Cypher query statement, supporting flexible graph database operations |
| `get_kg_statistics` | Obtain statistical information such as nodes, relationships, and type distribution of the knowledge graph |
| `get_entity_details` | Obtain detailed information and relationships of entities based on entity identifiers |
| `get_experiment_workflow` | Obtain the complete workflow of chemical experiments |

## Quick Start

### 1. Dependence
Recommended: Python 3.10+

Install `mcp`:

```bash
pip install mcp
```

### 2. Configuration
Define the Server client (Python):

```python
# Python ----------
import asyncio
import json

from mcp.client.streamable_http import streamablehttp_client
from mcp.client.session import ClientSession


class MultiDomainKGClient:
    def __init__(self, server_url: str = "https://scp.intern-ai.org.cn/api/v1/mcp/37/SciGraph"):
        self.server_url = server_url
        self.session = None

    async def connect(self):
        """建立连接并初始化会话"""
        print(
        )
        print("连接到 SciGraph SCP Server")
        print(
        )
        print(
        )
        print(
        )
        try:
            self.transport = streamablehttp_client(
                url=self.server_url,
                headers={"SCP-HUB-API-KEY": "sk-xxx"}
            )
            self.read, self.write, self.get_session_id = await self.transport.__aenter__()
            self.session_ctx = ClientSession(self.read, self.write)
            self.session = await self.session_ctx.__aenter__()
            await self.session.initialize()

            session_id = self.get_session_id()
            print(
                f"✓ 连接成功"
            )
            print(
            )
            print(
            )
            print(
            )
            return True
        except Exception as e:
            print(
            )
            import traceback
            traceback.print_exc()
            return False

    async def disconnect(self):
        """断开连接"""
        try:
            if self.session:
                await self.session_ctx.__aexit__(None, None, None)
            if hasattr(self, 'transport'):
                await self.transport.__aexit__(None, None, None)
            print("\n✓ 已断开连接\n")
        except Exception as e:
            print(
            )

    async def list_tools(self):
        """列出所有可用工具"""
        tools_list = await self.session.list_tools()
        print(
        )
        for i, tool in enumerate(tools_list.tools, 1):
            print(
            )
            if tool.description:
                desc_line = tool.description.split('\n')[0]
                print(
                )
        return tools_list.tools

    def parse_result(self, result):
        """解析 MCP 工具调用结果"""
        try:
            if hasattr(result, 'content') and result.content:
                content = result.content[0]
                if hasattr(content, 'text'):
                    return json.loads(content.text)
            return str(result)
        except Exception as e:
            return {"error":
            , "raw": str(result)}
```

## Usage
Taking **KnowDDI** as an example:

```python
# Python ----------
async def main():
    ## 客户端创建和连接
    client = MultiDomainKGClient()
    if not await client.connect():
        print("连接失败")
        return

    ## 示例1：获取知识图谱统计信息
    result = await client.session.call_tool(
        "get_kg_statistics",
        arguments={"kg_name": "KnowDDI"}  # 不指定 kg_name，返回所有图谱统计
    )
    stats = client.parse_result(result)
    print(stats)

    ## 示例2：查询 KnowDDI 实验的完整工作流
    result = await client.session.call_tool(
        "get_experiment_workflow",
        arguments={"experiment_id": "experiment_1"}
    )
    workflow = client.parse_result(result)
    print(workflow)

    ## 示例3：使用 Cypher 查询 KnowDDI 相关信息
    result = await client.session.call_tool(
        "query_cypher",
        arguments={
            "cypher": "MATCH (e:Experiment:KnowDDI) RETURN e.id as experiment_id",
            "kg_name": "KnowDDI",
            "limit": 5
        }
    )
    experiment_id = client.parse_result(result)
    print(experiment_id)

    ## 示例4：获取 KnowDDI 实体详情
    result = await client.session.call_tool(
        "get_entity_details",
        arguments={
            "entity_identifier": "experiment_1",
            "kg_name": "KnowDDI"
        }
    )
    entity = client.parse_result(result)
    print(entity)

    ## 客户端断开
    await client.disconnect()


if __name__ == '__main__':
    asyncio.run(main())
```

## How to use?
1. Install MCP SDK
   - Link: SDK documentation

   ```bash
   pip install mcp
   ```

2. Apply for an API Key
   - Application Portal: https://discovery-usercenter.intern-ai.org.cn/

3. Configuration Information

   Endpoint:

   ```text
   https://scp.intern-ai.org.cn/api/v1/mcp/37/SciGraph
   ```

   MCP server config:

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

## Aknowledgement & Reference
The contributors of this knowledge graph data is Baidu Research, Baidu Inc. Please cite the original paper when using this data:

Wang, Y., Yang, Z. & Yao, Q. Accurate and interpretable drug-drug interaction prediction enabled by knowledge subgraph learning. *Commun Med* 4, 59 (2024). https://doi.org/10.1038/s43856-024-00486-y
