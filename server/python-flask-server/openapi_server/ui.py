import datetime

from rl_reasoner.query import Query
##from openapi_server.models.response import Response  # noqa: E501
from openapi_server.models.message import Message
from openapi_server.models.node import Node
from openapi_server.models.edge import Edge
#from openapi_server.models.node_attribute import NodeAttribute
#from openapi_server.models.edge_attribute import EdgeAttribute
from openapi_server.models.knowledge_graph import KnowledgeGraph
from openapi_server.models.result import Result
#from openapi_server.models.message_terms import MessageTerms


def query(query_graph):
    q = Query()

    nodemap = {n.id: n for n in query_graph.nodes}

    if len(query_graph.edges) == 1:
        source_node = nodemap[query_graph.edges[0].source_id]
        target_node = nodemap[query_graph.edges[0].target_id]

        if source_node.type == "chemical_substance" and target_node.type == "disease" and target_node.curie == "?":
            query_entity = source_node.curie
            query_relation = query_graph.edges[0].type
            return q.query(int(query_entity), query_relation)
    
    msg =  "graph query not implemented"
    return( { "status": 501, "title": msg, "detail": msg, "type": "about:blank" }, 501 )
