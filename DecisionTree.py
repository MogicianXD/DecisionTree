import pandas as pd
import math
from graphviz import Digraph
from Tree import TreeNode


def _entropy(D: pd.DataFrame, class_name='class'):
    freqs = D[class_name].value_counts(normalize=True)
    return - sum([f * math.log2(f) for f in freqs])


def _gain(D, attr_name):
    g = _entropy(D)
    for a, df in D.groupby(attr_name):
        g -= len(df) / len(D) * _entropy(df)
    return g


class DecisionTree:
    def __init__(self, data, attr_dict=None, class_key='class', pre_pruning=False, valid=None):
        if attr_dict is None:
            attr_dict = dict()
            for attr in data.columns:
                if attr == class_key:
                    continue
                attr_dict[attr] = set(data[attr])
        self.attr_dict = attr_dict
        self.class_name = class_key
        self.pre_pruning = pre_pruning
        self.root = TreeNode(data={'classnum': data[self.class_name].value_counts()})
        self.root.name = self._generate(data, self.root, valid)

    def _choose_attr(self, data: pd.DataFrame):
        max_gain = 0
        gains = {}
        res = None
        for attr_name in data.columns:
            if attr_name == self.class_name:
                continue
            g = _gain(data, attr_name)
            gains[attr_name] = round(g, 2)
            if g > max_gain:
                max_gain = g
                res = attr_name
        return res, gains

    def _generate(self, data, parent: TreeNode, valid=None):
        classes = data[self.class_name].unique()
        if len(classes) == 1:
            return classes.item()
        if data.shape[1] == 1:
            return data[self.class_name].value_counts().index[0]

        choice, parent.data['gains'] = self._choose_attr(data)

        groups = data.groupby(choice)
        empty_a = self.attr_dict[choice] - set(data[choice])
        classnum = data[self.class_name].value_counts()
        majority = classnum.index[0]

        if self.pre_pruning:
            acc2 = (valid[self.class_name] == majority).sum()
            acc = 0
            for a, df in groups:
                m = df[self.class_name].value_counts().index[0]
                acc += ((valid[choice] == a) & (valid[self.class_name] == m)).sum()
            if empty_a:
                for a in empty_a:
                    acc += ((valid[choice] == a) & (valid[self.class_name] == majority)).sum()
            if acc2 > acc:
                print(acc2, acc)
                return majority

        for a, df in groups:
            child = parent.add_child(edge=a, data={'classnum': df[self.class_name].value_counts()})
            child.name = self._generate(df.drop(choice, axis=1), child,
                                        valid[valid[choice]==a].drop(choice, axis=1) if valid is not None else None)

        if empty_a:
            for a in empty_a:
                child = parent.add_child(edge=a)
                child.name = majority

        return choice

    def predict(self, data):
        p = self.root
        while not p.is_leaf():
            p = p.child[data[p.name]]
        return p.name


    def post_pruning(self, valid):
        self._post_pruning(self.root, valid)

    def _post_pruning(self, node, valid):
        consider = True
        for edge, child in node.child.items():
            if not child.is_leaf():
                if not self._post_pruning(child, valid[valid[node.name] == edge]):
                    consider = False
        if consider:
            acc = 0
            for edge, child in node.child.items():
                acc += len((valid[node.name] == edge) & (valid[self.class_name] == child.name).sum())
            # if be pruned
            majority = node.data['classnum'].index[0]
            if (valid[self.class_name] == majority).sum() >= acc:
                node.child = dict()
                node.name = majority
                return True
        return False

    def plot(self):
        print('root')
        self._plot(self.root)

    def _plot(self, node, deepin=0):
        prefix = '|   '
        for edge, child in node.child.items():
            if not child.is_leaf():
                print(prefix * deepin + '|——' + node.name + '=' + edge)
                self._plot(child, deepin=deepin + 1)
            else:
                if deepin == 0:
                    print('|——' + node.name + '=' + edge + '————' +child.name)
                else:
                    # prefix += '|'
                    print(prefix * deepin + '|——' + node.name + '=' + edge + '————' +child.name)

    def draw(self):
        '''BSF'''
        dot = Digraph(comment="Decision Tree")
        lis = []
        lis.append(self.root)
        while len(lis) != 0:
            node = lis.pop(0)
            dot.node(str(hash(node)), label=node.name + '\n' + str(node.data['gains']), fontname='utf-8', shape='rect')
            for edge, child in node.child.items():
                if child.is_leaf():
                    dot.node(str(hash(child)), child.name, fontname='utf-8')
                else:
                    dot.node(str(hash(child)), child.name + '\n' + str(child.data['gains']), fontname='utf-8', shape='rect')
                    lis.append(child)
                dot.edge(str(hash(node)), str(hash(child)), label=edge, fontname='utf-8')
        dot.view()

