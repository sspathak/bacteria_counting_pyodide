#####################
# Frontier object
class Frontier:
    def __init__(self):
        self.record = {}
        self.frontier = []

    def add(self, item):
        self.frontier.append(item)
        self.record[str(item)] = True

    def pop(self):
        item = self.frontier[0]
        self.record[str(item)] = False
        self.frontier = self.frontier[1:]
        return item
    
    def contains(self, item):
        key = str(item)
        return (key in self.record) and self.record[key]

    def isEmpty(self):
        return len(self.frontier) == 0
    