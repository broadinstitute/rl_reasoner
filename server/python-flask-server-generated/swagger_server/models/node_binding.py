# coding: utf-8

from __future__ import absolute_import
from datetime import date, datetime  # noqa: F401

from typing import List, Dict  # noqa: F401

from swagger_server.models.base_model_ import Model
from swagger_server.models.one_of_node_binding_kg_id import OneOfNodeBindingKgId  # noqa: F401,E501
from swagger_server import util


class NodeBinding(Model):
    """NOTE: This class is auto generated by the swagger code generator program.

    Do not edit the class manually.
    """
    def __init__(self, qg_id: str=None, kg_id: OneOfNodeBindingKgId=None):  # noqa: E501
        """NodeBinding - a model defined in Swagger

        :param qg_id: The qg_id of this NodeBinding.  # noqa: E501
        :type qg_id: str
        :param kg_id: The kg_id of this NodeBinding.  # noqa: E501
        :type kg_id: OneOfNodeBindingKgId
        """
        self.swagger_types = {
            'qg_id': str,
            'kg_id': OneOfNodeBindingKgId
        }

        self.attribute_map = {
            'qg_id': 'qg_id',
            'kg_id': 'kg_id'
        }
        self._qg_id = qg_id
        self._kg_id = kg_id

    @classmethod
    def from_dict(cls, dikt) -> 'NodeBinding':
        """Returns the dict as a model

        :param dikt: A dict.
        :type: dict
        :return: The NodeBinding of this NodeBinding.  # noqa: E501
        :rtype: NodeBinding
        """
        return util.deserialize_model(dikt, cls)

    @property
    def qg_id(self) -> str:
        """Gets the qg_id of this NodeBinding.

        Query-graph node id, i.e. the `node_id` of a QNode  # noqa: E501

        :return: The qg_id of this NodeBinding.
        :rtype: str
        """
        return self._qg_id

    @qg_id.setter
    def qg_id(self, qg_id: str):
        """Sets the qg_id of this NodeBinding.

        Query-graph node id, i.e. the `node_id` of a QNode  # noqa: E501

        :param qg_id: The qg_id of this NodeBinding.
        :type qg_id: str
        """
        if qg_id is None:
            raise ValueError("Invalid value for `qg_id`, must not be `None`")  # noqa: E501

        self._qg_id = qg_id

    @property
    def kg_id(self) -> OneOfNodeBindingKgId:
        """Gets the kg_id of this NodeBinding.

        One or more knowledge-graph node ids, i.e. the `id` of a KNode  # noqa: E501

        :return: The kg_id of this NodeBinding.
        :rtype: OneOfNodeBindingKgId
        """
        return self._kg_id

    @kg_id.setter
    def kg_id(self, kg_id: OneOfNodeBindingKgId):
        """Sets the kg_id of this NodeBinding.

        One or more knowledge-graph node ids, i.e. the `id` of a KNode  # noqa: E501

        :param kg_id: The kg_id of this NodeBinding.
        :type kg_id: OneOfNodeBindingKgId
        """
        if kg_id is None:
            raise ValueError("Invalid value for `kg_id`, must not be `None`")  # noqa: E501

        self._kg_id = kg_id
