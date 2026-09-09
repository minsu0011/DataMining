import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def load(relative):
    spec = importlib.util.spec_from_file_location(relative.replace('/', '_'), ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def test_apriori_synthetic_support():
    mod = load('project1/new.py')
    frequent = mod.apriori([[1, 2], [1, 2], [1, 3], [2, 3]], .5)
    assert frozenset([1, 2]) in frequent
    assert abs(frequent[frozenset([1, 2])] - .5) < 1e-12

def test_tree_seen_categories():
    mod = load('project2/new.py')
    data = [['sun'], ['sun'], ['rain'], ['rain']]
    tree = mod.buildTree(data, ['yes', 'yes', 'no', 'no'], [0])
    assert mod.classify(tree, ['sun']) == 'yes'
    assert mod.classify(tree, ['rain']) == 'no'
    assert abs(mod.entropy(['yes', 'no']) - 1) < 1e-12
