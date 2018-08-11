import numpy as np
import networkx as nx
import csv

class KGEnvironment():
    def __init__(self,
                 query_entity,
                 query_relation,
                 target,
                 graph_file):

        self.graph, self.entities, self.relations = self.from_csv(graph_file)

        self.observation_dim = (2,)
        self.spec = {"id": "kgenv"}

        self.query_entity = self.entities[query_entity]
        self.query_relation = self.relations[query_relation]

        self.target = self.entities[target]
        self.target_found = False

        self.reset()

    def from_csv(self, graph_file):
        graph = nx.MultiDiGraph()
        entities = {}
        relations = {'NO_OP':0, 'DUMMY_RELATION':1}
        with open(graph_file) as f:
            csv_file = csv.reader(f, delimiter='\t')
            entity_counter = len(entities)
            relation_counter = len(relations)
            for line in csv_file:
                if line[0] in entities:
                    start = entities[line[0]]
                else:
                    entities[line[0]] = entity_counter
                    start = entity_counter
                    entity_counter = entity_counter + 1

                if line[2] in entities:
                    end = entities[line[2]]
                else:
                    entities[line[2]] = entity_counter
                    end = entity_counter
                    entity_counter = entity_counter + 1

                if line[1] in relations:
                    relation = relations[line[1]]
                else:
                    relations[line[1]] = relation_counter
                    relation = relation_counter
                    relation_counter = relation_counter + 1

                graph.add_node(start)
                graph.add_node(end)
                graph.add_edge(start, end, key=relation)
        return graph, entities, relations

    def get_graph(self):
        return self.graph

    def reset(self):
        print("reset")
        self.current_relation = self.relations['DUMMY_RELATION']
        self.current_entity = self.query_entity
        self.next_actions = self.generate_next_actions()
        return np.array(self.get_state())

    def get_state(self):
        return np.array((self.current_relation, self.current_entity))

    def get_query(self):
        return np.array((self.query_entity, self.query_relation))

    def step(self, action_idx):
        action = self.next_actions[action_idx]
        print(action)
        if self.graph.has_edge(self.current_entity, action[1], key=action[0]):
            self.current_relation = action[0]
            self.current_entity = action[1]
            self.next_actions = self.generate_next_actions()

        if self.target_found:
            return np.array(self.get_state()), 0, True, {}
        else:
            if self.current_entity == self.target:
                self.target_found = True
                return np.array(self.get_state()), 1, True, {}
            else:
                return np.array(self.get_state()), 0, False, {}

    def generate_next_actions(self):
        next_actions = [(e[2],e[1]) for e in self.graph.edges(self.current_entity, keys=True)]
        return next_actions

    def get_available_actions(self):
        return self.next_actions
