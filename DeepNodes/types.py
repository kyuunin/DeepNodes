  
class Type:
    def __init__(self, name):
        self.name = name
    def __and__(self,other):
        return Union_Type(self,other)
    def __or__(self,other):
        return Option_Type(self,other)
    def __repr__(self):
        return self.name
class Union_Type(Type):
    def __init__(self,*args):
        self.args = [x for xs in args for x in (xs.args if type(xs) is Union_Type else (xs,))]
    def __repr__(self):
        return "("+",".join(map(str,self.args))+")"
class Option_Type(Type):
    def __init__(self,*args):
        self.args = {x for xs in args for x in (xs.args if type(xs) is Option_Type else (xs,))}
    def __repr__(self):
        return "("+"|".join(map(str,self.args))+")"


none = Type("none")
auto = Type("auto")
Auto = "auto"
integer = Type("integer")
string = Type("string")
tensor = Type("tensor")
activation = Type("activation")
