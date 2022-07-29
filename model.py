
import torch.nn as nn 


class ChatROBOT(nn.Module):
     def __init__(self, input, hidden, output):
          super(ChatROBOT, self).__init__()
          self.input_layer = nn.Linear(input, hidden) 
          self.first_layer = nn.Linear(hidden, hidden) 
          self.second_layer = nn.Linear(hidden, hidden)
          self.third_layer = nn.Linear(hidden, hidden) 
          self.output_layer = nn.Linear(hidden, output)
          self.relu = nn.ReLU()

     def forward(self, x):
          x = self.relu( self.input_layer(x) )
          x = self.relu( self.first_layer(x) )
          x = self.relu( self.second_layer(x) )
          x = self.relu( self.third_layer(x) )
          x = self.output_layer(x)
          return x