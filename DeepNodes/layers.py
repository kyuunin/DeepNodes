from torch.nn import Module
import torch.nn.functional as F

layers = dict()

class Layer(Module):
    pass
    
class Layer_Factory():
    def __init__(self,layer):
        self.data = {k:v[0] for k,v in layer.args.items()}
        self.layer = layer
        self._input_shape = {k:[None]*(v if type(v) is int else v[0]) for k,v in layer.input_dim._asdict().items()}
        self._output_shape = {k:[None]*(v if type(v) is int else v[0]) for k,v in layer.output_dim._asdict().items()}
        self.data["name"] = layer.__name__
        self.data["input_shape"] =  layer.inputs(**self._input_shape)
        self.data["output_shape"] = layer.outputs(**self._output_shape)
        
    def __call__(self):
        return self.layer(**self.data)
        
    def set_input_shape(self, key, *values):
        dim = getattr(self.layer.input_dim,key)
        if type(dim) == int:
            if len(values) != dim:
                raise ValueError
        elif len(values) not in range(*dim):
            raise ValueError
        self._input_shape[key]=values
        self.data["input_shape"] =  self.layer.inputs(**self._input_shape)
        
    def set_output_shape(self, key, *values):
        dim = getattr(self.layer.output_dim,key)
        if type(dim) == int:
            if len(values) != dim:
                raise ValueError
        elif len(values) not in range(*dim):
            raise ValueError
        self._output_shape[key]=values
        self.data["output_shape"] =  self.layer.outputs(**self._output_shape)
        
    def __getattribute__(self,key):
        if key in super().__getattribute__("data"):
            return super().__getattribute__("data")[key]
        else:
            return super().__getattribute__(key)
            
    def __setattr__(self,key,value):
        if key in {"input_shape","output_shape"}:
            raise  AttributeError
        if key == "data":
            super().__setattr__(key,value)
        if key in self.data:
            self.data[key] = value
        else:
            super().__setattr__(key,value)
      
