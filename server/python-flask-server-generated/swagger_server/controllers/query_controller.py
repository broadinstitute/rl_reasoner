import connexion
import six

from swagger_server.models.message import Message  # noqa: E501
from swagger_server.models.query import Query
from swagger_server import util

from swagger_server import ui

def query(request_body):  # noqa: E501
    """Query reasoner via one of several inputs

     # noqa: E501

    :param body: Query information to be submitted
    :type body: dict | bytes

    :rtype: Message
    """
    if connexion.request.is_json:
        body = Query.from_dict(connexion.request.get_json())  # noqa: E501
        print(body)
        return ui.query(body.message.query_graph)

    return({"status": 400, "title": "body content not JSON", "detail": "Required body content is not JSON", "type": "about:blank"}, 400)
