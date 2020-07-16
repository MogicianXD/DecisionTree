import numpy as np
import pandas as pd
from DecisionTree import DecisionTree
from sklearn.tree import DecisionTreeClassifier

DecisionTreeClassifier()

np.random.seed(233)


df = pd.read_csv('car.data', header=None,
                 names=['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class'])
class_key = 'class'
attr_dict = dict()
for attr in df.columns:
    if attr == class_key:
        continue
    attr_dict[attr] = set(df[attr])
indice = np.random.permutation(len(df))

train_data = df.loc[indice[:int(len(df) * 0.6)]]
valid_data = df.loc[indice[int(len(df) * 0.6): int(len(df) * 0.8)]]
test_data = df.loc[indice[int(len(df) * 0.8):]]

# train_data = df.loc[indice[:20]]
# train_data.to_excel('twenty.xlsx')
# test_data = df.loc[indice[20:70]]

tree = DecisionTree(train_data, attr_dict, class_key, pre_pruning=False, valid=valid_data)
tree.post_pruning(valid_data)
tree.plot()
tree.draw()

classes = df['class'].unique().tolist()
class_dict = {c: i for i, c in enumerate(classes)}
confusion_mat = np.zeros((len(classes), len(classes)), dtype=np.int)

# test_data = valid_data
for id, row in test_data.iterrows():
    pred = tree.predict(row)
    confusion_mat[class_dict[row['class']]][class_dict[pred]] += 1
pd.DataFrame(confusion_mat).to_excel('confusion_matrix.xlsx')

print()
f1s, TPs, Ps = [], [], []
eps = 1e-10
for i, c in enumerate(classes):
    TP = confusion_mat[i][i]
    T = confusion_mat[:, i].sum()
    P = confusion_mat[i, :].sum()
    TN = len(test_data) - T - P + TP
    acc = (TP + TN) / len(test_data)
    precison = TP / (T + eps)
    recall = TP / (P + eps)
    score_F1 = 2 * precison * recall / (precison + recall + eps)
    print('{}:{}\t准确率 {:.2%}\t精度 {:.2%}\t召回率 {:.2%}\tF1度量 {:.2%}'
          .format(c, ' ' * (5 - len(c)), acc, precison, recall, score_F1))
    f1s.append(score_F1)
    TPs.append(TP)
    Ps.append(P)

macro_F1 = sum(f1s) / len(f1s)
micro_F1 = sum(TPs) / (sum(Ps) + eps)
print('macro_F1 {:.2%}'.format(macro_F1))
print('micro_F1 {:.2%}'.format(micro_F1))


