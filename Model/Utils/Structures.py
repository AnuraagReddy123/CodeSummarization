class Node:
    def __init__(self, id:int):
        self.id = id
        self.children = []
        self.emb = None
        self.h = None
        self.c = None
        self.out = None
    
    def add_child(self, child):
        self.children.append(child)
        