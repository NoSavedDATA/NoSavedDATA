from torch import nn
import inspect


# Hyper Parameters
# automatically saves all arguments of the inherited class __init__
class Hypers(object):
    def __init__(self, depth=2):
        super().__init__()
        self.save_hypers(depth)
    
    def save_hypers(self, depth=1, ignore=[]):
      """Save function arguments into class attributes."""

      #f_back: frame caller
      #frame: table of local variablies to the frame's function
      frame = inspect.currentframe()
      for d in range(depth):
        frame = frame.f_back
      _, _, _, local_vars = inspect.getargvalues(frame)
      #takes the arguments of the function which called this save_hypers function
      #it can backtrack functions according to the depth argument

      self.hparams = {k:v for k, v in local_vars.items()
          if k not in set(ignore+['self']) and not k.startswith('_')}
      for k, v in self.hparams.items():
          setattr(self, k, v)

class nsd_Module(Hypers, nn.Module):
    def __init__(self):
        super().__init__(depth=3)