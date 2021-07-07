import torch
import torch.nn as nn


class LinearBlock(nn.Module):
    
    def __init__(self, 
                 input_size, 
                 output_size, 
                 use_batch_norm=True, 
                 dropput_p = 0.4):
        self.input_size = input_size
        self.output_size = output_size
        self.use_batch_norm = use_batch_norm
        self.dropput_p = dropput_p
        
        super().__init__()
        
        def get_regularizer(use_batch_norm, size):
            return nn.BatchNorm1d(size) if use_batch_norm else nn.Dropout(dropput_p)
        
        self.linearblock = nn.Sequential(
            nn.Linear(input_size, output_size, bias=False),
            nn.LeakyReLU(),
            get_regularizer(use_batch_norm, output_size)
        )
        
    def forward(self, x):
        # |x| = (batch_size, input_size)
        y = self.linearblock(x)
        # |y| = (batch_size, output_size)
        
        return y
    

class FCL_Model(nn.Module):
    
    def __init__(self, 
                 input_size, 
                 output_size, 
                 hidden_sizes=[500, 400, 300, 200, 100],
                 use_batch_norm=True,
                 dropout_p=0.3):
        
        super().__init__()
        
        assert len(hidden_sizes) > 0, "You need to specify hidden layers"
        
        last_hidden_size = input_size
        blocks = []
        
        for hiden_size in hidden_sizes:
            blocks += [LinearBlock(last_hidden_size,
                        hiden_size,
                        use_batch_norm,
                        dropout_p
                        )]
            last_hidden_size = hiden_size
        
        self.layers = nn.Sequential(
            *blocks,
            nn.Linear(last_hidden_size, output_size),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # |x| = (batch_size, input_size)      
        y = self.layers(x)
        # |y| = (batch_size, output_size)
        
        return y

if __name__ == '__main__':
    model = FCL_Model(input_size=6, 
                      output_size=1, 
                      hidden_sizes=[100, 100],
                      use_batch_norm=False)
    print(model)