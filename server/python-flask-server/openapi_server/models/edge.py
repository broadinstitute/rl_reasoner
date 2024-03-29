# coding: utf-8

from __future__ import absolute_import
from datetime import date, datetime  # noqa: F401

from typing import List, Dict  # noqa: F401

from openapi_server.models.base_model_ import Model
from openapi_server import util


class Edge(Model):
    """NOTE: This class is auto generated by OpenAPI Generator (https://openapi-generator.tech).

    Do not edit the class manually.
    """

    def __init__(self, id=None, type=None, source_id=None, target_id=None):  # noqa: E501
        """Edge - a model defined in OpenAPI

        :param id: The id of this Edge.  # noqa: E501
        :type id: str
        :param type: The type of this Edge.  # noqa: E501
        :type type: str
        :param source_id: The source_id of this Edge.  # noqa: E501
        :type source_id: str
        :param target_id: The target_id of this Edge.  # noqa: E501
        :type target_id: str
        """
        self.openapi_types = {
            'id': str,
            'type': str,
            'source_id': str,
            'target_id': str
        }

        self.attribute_map = {
            'id': 'id',
            'type': 'type',
            'source_id': 'source_id',
            'target_id': 'target_id'
        }

        self._id = id
        self._type = type
        self._source_id = source_id
        self._target_id = target_id

    @classmethod
    def from_dict(cls, dikt) -> 'Edge':
        """Returns the dict as a model

        :param dikt: A dict.
        :type: dict
        :return: The Edge of this Edge.  # noqa: E501
        :rtype: Edge
        """
        return util.deserialize_model(dikt, cls)

    @property
    def id(self):
        """Gets the id of this Edge.

        Local identifier for this node which is unique within this KnowledgeGraph, and perhaps within the source reasoner's knowledge graph  # noqa: E501

        :return: The id of this Edge.
        :rtype: str
        """
        return self._id

    @id.setter
    def id(self, id):
        """Sets the id of this Edge.

        Local identifier for this node which is unique within this KnowledgeGraph, and perhaps within the source reasoner's knowledge graph  # noqa: E501

        :param id: The id of this Edge.
        :type id: str
        """
        if id is None:
            raise ValueError("Invalid value for `id`, must not be `None`")  # noqa: E501

        self._id = id

    @property
    def type(self):
        """Gets the type of this Edge.

        A relation, i.e. child of related_to (snake_case)  # noqa: E501

        :return: The type of this Edge.
        :rtype: str
        """
        return self._type

    @type.setter
    def type(self, type):
        """Sets the type of this Edge.

        A relation, i.e. child of related_to (snake_case)  # noqa: E501

        :param type: The type of this Edge.
        :type type: str
        """

        self._type = type

    @property
    def source_id(self):
        """Gets the source_id of this Edge.

        Corresponds to the @id of source node of this edge  # noqa: E501

        :return: The source_id of this Edge.
        :rtype: str
        """
        return self._source_id

    @source_id.setter
    def source_id(self, source_id):
        """Sets the source_id of this Edge.

        Corresponds to the @id of source node of this edge  # noqa: E501

        :param source_id: The source_id of this Edge.
        :type source_id: str
        """
        if source_id is None:
            raise ValueError("Invalid value for `source_id`, must not be `None`")  # noqa: E501

        self._source_id = source_id

    @property
    def target_id(self):
        """Gets the target_id of this Edge.

        Corresponds to the @id of target node of this edge  # noqa: E501

        :return: The target_id of this Edge.
        :rtype: str
        """
        return self._target_id

    @target_id.setter
    def target_id(self, target_id):
        """Sets the target_id of this Edge.

        Corresponds to the @id of target node of this edge  # noqa: E501

        :param target_id: The target_id of this Edge.
        :type target_id: str
        """
        if target_id is None:
            raise ValueError("Invalid value for `target_id`, must not be `None`")  # noqa: E501

        self._target_id = target_id
