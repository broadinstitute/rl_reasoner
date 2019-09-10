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

    if length(query_graph.edges) == 1:
        source_node = query_graph.nodes[query_graph.edges[0].source_id]
        target_node = query_graph.nodes[query_graph.edges[0].target_id]

        if source_node.type == "chemcial_substance" & target_node.type == "disease" & target_node.curie == "?":
            query_entity = source_node.curie
            query_relation = query_graph.edges[0].type
            return q.query(query_entity, query_relation)
    
    msg =  "graph query not implemented"
    return( { "status": 501, "title": msg, "detail": msg, "type": "about:blank" }, 501 )
