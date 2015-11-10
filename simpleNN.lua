require 'nn'

-- Create new neural network
mynn = nn.Sequential() -- Creates a sequential model. Alternative : Parallel, Concat

-- Add Linear layer to neural network
mynn:add(nn.Linear(5,5)) --Add one linear layer with input size 10 and output size 15
mynn:add(nn.Linear(5,5)) -- Order in which the layers are added is important

-- Update weights
function gradUpdate(mlp, x, y, criterion, learningRate)
   local pred = mlp:forward(x)
   local err = criterion:forward(pred, y)
   local gradCriterion = criterion:backward(pred, y)
   mlp:zeroGradParameters()
   mlp:backward(x, gradCriterion)
   mlp:updateParameters(learningRate)
   return err
end 

-- Training data
y = torch.Tensor(5)
for i = 1, 10000 do
   x = torch.rand(5)
   y:copy(x)
   y:mul(math.pi)
   err = gradUpdate(mynn, x, y, nn.MSECriterion(), 0.01)
end

-- Test data
input = torch.Tensor({1,2,3,4,5})
output = mynn:forward(input)
print(output)
