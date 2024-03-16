from torch import nn
import inspect


# Hyper Parameters
# automatically saves all arguments of the inherited class __init__
class Hypers: # Sorcery
    def __init__(self, max_depth=2, **kwargs):
        super().__init__(**kwargs)
        self.save_hypers(max_depth)
    
    def save_hypers(self, max_depth, ignore=[]):
      """Save function arguments into class attributes."""

      #f_back: frame caller
      #frame: table of local variablies to the frame's function
      seen_init=False
      frame = inspect.currentframe()
      for d in range(max_depth):
          
          frame = frame.f_back
          
          if frame.f_back and frame.f_back.f_code.co_name == "__init__":
              seen_init=True
              
          if seen_init and frame.f_back.f_code.co_name != "__init__":
              break
            
      _, _, _, local_vars = inspect.getargvalues(frame)
      #takes the arguments of the function which called this save_hypers function
      #it can backtrack functions according to the depth argument

      self.hparams = {k:v for k, v in local_vars.items()
          if k not in set(ignore+['self']) and not k.startswith('_')}
      for k, v in self.hparams.items():
          setattr(self, k, v)


# ALLWAYS PUT HYPERS TO THE LEFT
class nsd_Module(Hypers, nn.Module):
    def __init__(self):
        super().__init__(max_depth=3)