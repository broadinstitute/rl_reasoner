import numpy as np
import networkx as nx
import csv
from reasoner.knowledge_graph.KnowledgeGraph import KnowledgeGraph


class KGEnvironment():
    def __init__(self,
                 graph,
                 num_entities,
                 num_relations,
                 queries,
                 spec):

        self.graph = graph
        self.num_entities = num_entities
        self.num_relations = num_relations

        self.queries = queries
        self.observation_dim = (2,)
        self.spec = spec
        self.target_found = False
        self.reset()

    def get_num_entities(self):
        return self.num_entities 

    def get_num_relations(self):
        return self.num_relations

    def get_graph(self):
        return self.graph

    def get_state(self):
        return np.array((self.current_relation, self.current_entity))

    def get_query(self):
        return np.array((self.query_entity, self.query_relation))

    def get_available_actions(self):
        return self.next_actions

    def reset(self):
        entity_idx = np.random.randint(0, len(self.queries))
        relation_idx = np.random.randint(0, len(self.queries[entity_idx]))
        self.query_entity = self.queries.keys()[entity_idx]
        self.query_relation = self.queries[entity_idx].keys()[relation_idx]
        self.targets = self.queries[entity_idx][relation_idx]

        self.target_found = False
        self.current_relation = self.relations['DUMMY_RELATION']
        self.current_entity = self.query_entity
        self.next_actions = self.generate_next_actions()
        return np.array(self.get_state()), np.array(self.get_available_actions()), self.query_relation




class NxEnvironment(KGEnvironment):
    def __init__(self, graph_file, query_file):
        graph, self.entities, self.relations = self.from_csv(graph_file)
        queries = self.read_queries(query_file)
        spec = {"id": "nxenv"}
        super().__init__(graph, len(self.entities), len(self.relations), queries, spec)

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

    def read_queries(self, query_file):
        with open(query_file) as f:
            queries = list()
            csv_file = csv.reader(f, delimiter='\t')
            for line in csv_file:
                query_entity = self.entities[line[0]]
                query_relation = self.relations[line[1]]
                query_target = self.entities[line[2]]
                if query_entity not in queries:
                    queries[query_entity] = dict()
                if query_relation not in queries[query_entity]:
                    queries[query_entity][query_relation] = list()
                queries[query_entity][query_relation].add(query_target)
        return queries

    def step(self, action_idx):
        action = self.next_actions[action_idx]
        if action[0] == self.relations['NO_OP']:
            self.current_relation = action[0]
        elif self.graph.has_edge(self.current_entity, action[1], key=action[0]):
            self.current_relation = action[0]
            self.current_entity = action[1]
            self.next_actions = self.generate_next_actions()

        if self.target_found:
            return np.array(self.get_state()), np.array(self.get_available_actions()), 0, True, {}
        else:
            if self.current_entity in self.targets:
                self.target_found = True
                return np.array(self.get_state()), np.array(self.get_available_actions()), 1, True, {}
            else:
                return np.array(self.get_state()), np.array(self.get_available_actions()), 0, False, {}

    def generate_next_actions(self):
        next_actions = [(e[2],e[1]) for e in self.graph.edges(self.current_entity, keys=True)]
        next_actions.append((self.relations['NO_OP'], self.current_entity)) # add NO-OP
        return next_actions


class Neo4jEnvironment(KGEnvironment):
    def __init__(self, query_file, entities):
        graph = KnowledgeGraph()
        #num_entities = graph.get_num_nodes()
        self.entity_list = entities
        self.entities = {item:idx for idx,item in enumerate(entities)}
        predicates = [record["predicate"] for record in graph.get_predicates()]
        self.relations = {item:idx for idx,item in enumerate(['NO_OP', 'DUMMY_RELATION'] + predicates)}
        queries = self.read_queries(query_file)
        print(queries)
        spec = {"id": "neoenv"}
        super().__init__(graph, len(self.entities), len(self.relations), queries, spec)

    def read_queries(self, query_file):
        with open(query_file) as f:
            queries = dict()
            csv_file = csv.reader(f, delimiter='\t')
            for line in csv_file:
                query_entity = self.entities[int(line[0])]
                query_relation = self.relations[line[1]]
                query_target = self.entities[int(line[2])]
                if query_entity not in queries:
                    queries[query_entity] = dict()
                if query_relation not in queries[query_entity]:
                    queries[query_entity][query_relation] = list()
                queries[query_entity][query_relation].add(query_target)
        return queries

    def step(self, action_idx):
        action = self.next_actions[action_idx]
        if action[0] == self.relations['NO_OP']:
            self.current_relation = action[0]
        #elif self.graph.has_edge(start=self.current_entity, end=action[1], predicate=action[0]):
        else:
            self.current_relation = action[0]
            self.current_entity = action[1]
            self.next_actions = self.generate_next_actions()

        if self.target_found:
            return np.array(self.get_state()), np.array(self.get_available_actions()), 0, True, {}
        else:
            if self.current_entity in self.targets:
                self.target_found = True
                return np.array(self.get_state()), np.array(self.get_available_actions()), 1, True, {}
            else:
                return np.array(self.get_state()), np.array(self.get_available_actions()), 0, False, {}

    def generate_next_actions(self):
        out_neighbors = self.graph.get_out_neighbors(self.entity_list[self.current_entity])
        
        if self.current_entity == self.query_entity:
            next_actions = [(self.relations[n['predicate']], self.entities[n['node_id']]) for n in out_neighbors if self.relations[n['predicate']] != self.query_relation and self.entities[n['node_id']] != self.target]
        else:
            next_actions = [(self.relations[n['predicate']], self.entities[n['node_id']]) for n in out_neighbors]
        next_actions.append((self.relations['NO_OP'], self.current_entity)) # add NO-OP
        print(self.current_entity, self.query_entity, self.query_relation, self.target)
        #print(next_actions)
        return next_actions

    def get_available_actions(self):
        return self.next_actions
