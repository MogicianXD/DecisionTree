
class TreeNode(object):
    def __init__(self, name=None, parent=None, child=None, data=None):
        super(TreeNode, self).__init__()
        self.name = name
        self.parent = parent
        self.child = child if child else dict()
        self.data = data

    def get_child(self, name, defval=None):
        return self.child.get(name, defval)

    def add_child(self, name=None, edge=None, data=None, obj=None):
        if obj and not isinstance(obj, TreeNode):
            raise ValueError('TreeNode only add another TreeNode obj as child')
        if not (name or edge):
            raise ValueError('TreeNode must have a name or edge_name to put in childs')
        if obj is None:
            obj = TreeNode(name, data=data)
        obj.parent = self
        self.child[edge if edge else name] = obj
        return obj

    def del_child(self, name):
        if name in self.child:
            del self.child[name]

    def find_child(self, path, create=False):
        # convert path to a list if input is a string
        path = path if isinstance(path, list) else path.split()
        cur = self
        for sub in path:
            # search
            obj = cur.get_child(sub)
            if obj is None and create:
                # create new node if need
                obj = cur.add_child(sub)
            # check if search done
            if obj is None:
                break
            cur = obj
        return obj

    def is_leaf(self):
        return len(self.child) == 0
