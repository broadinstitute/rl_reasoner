import connexion
import six

from swagger_server.models.message import Message  # noqa: E501
from swagger_server import util


def query(body):  # noqa: E501
    """Query reasoner via one of several inputs

     # noqa: E501

    :param body: Query information to be submitted
    :type body: dict | bytes

    :rtype: Message
    """
    if connexion.request.is_json:
        body = Dict.from_dict(connexion.request.get_json())  # noqa: E501
    return 'do some magic!'
