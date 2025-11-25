# Included to bound each bateria by a box
class Bounds:
    def __init__(self):
        self.init = False
        self.left = -1
        self.right = -1
        self.top = -1
        self.bottom = -1

    def add(self, coord):
        # coord in form (x, y)
        x, y = coord

        # Init part
        if not self.init:
            self.left = y
            self.right = y
            self.top = x
            self.bottom = x
            self.init = True
        
        # Update boundaries
        else:
            self.left = min(self.left, y)
            self.right = max(self.right, y)
            self.top = min(self.top, x)
            self.bottom = max(self.bottom, x)
    
    # shape of the bound
    def suggest_shape(self):
        return (self.bottom - self.top + 1, self.right - self.left + 1)

    # return informations needed for the bound
    def info(self):
        return (self.left, self.right, self.top, self.bottom)

    # Convert to string (easier to print)
    def __str__(self):
        return "left: {} right: {} top: {} bottom: {}".format(self.left, self.right, self.top, self.bottom)
