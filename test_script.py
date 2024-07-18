from torch.nn import ModuleList
import torch 

@torch.jit.interface
class ModuleInterface(torch.nn.Module):
    def forward(self, input: torch.Tensor) -> torch.Tensor: # `input` has a same name in Sequential forward
        pass

class Mine(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([torch.nn.Linear(32, 32), torch.nn.Linear(32, 32)])

    def forward(self, x: torch.Tensor, c: int):
        # for i in range(c):
        #     submodule: ModuleInterface = self.layers[i]
        #     result = submodule.forward(x)
        #     return result 
        res = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            res.append(x)
            

torch.jit.script(Mine())