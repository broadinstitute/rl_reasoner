import pytest
import csv
import numpy as np
from rl_reasoner import environment

@pytest.fixture(scope='session')
def graph_file(tmpdir_factory):
    data = [["Lisa", "sister_of", "Bart"],
            ["Maggie", "sister_of", "Bart"],
            ["Homer", "father_of", "Lisa"],
            ["Marge", "mother_of", "Lisa"]]

    filename = str(tmpdir_factory.mktemp('data').join('data.tsv'))
    with open(filename, 'w') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter='\t')
        for line in data:
            csvwriter.writerow(line)
    return filename


def test_environment(graph_file):
    env = environment.KGEnvironment("Lisa", "sister_of", "Bart", graph_file)
    state = env.get_state()
    assert state.shape == (2,)
    assert state[0] == 1
    assert state[1] == 0
    assert env.get_available_actions() == [(2,1)]

    next_action = env.get_available_actions()[0]
    state, reward, done, info = env.step(next_action)
    assert reward == 1
    assert done == True
