-- Code for building and training a simple RNN

-- Index Input	Output    
-- 6     1234		5
-- 7     2345		6
-- 8     3456		7
-- 9     4567		8
-- 10    5678		9
-- 1     6789		0
-- 2     7890		1
-- 3     8901		2
-- 4     9012		3
-- 5     0123		4

inputs = {}
outputs = {}
trainset = {}
for i=0,8 do
    local ip = {i+1}
    local op = {i+2}
    inputs[i+1] = ip
    outputs[i+1] = op
end
    inputs[10] = {10}
    outputs[10] = {1}
trainset[inputs] = inputs
trainset[outputs] = outputs

require 'nn'
require 'rnn'

lm = nn.Sequential()
lm:add(nn.LookupTable(10,6))  -- Input is index 1 to 10, output is vector of 4
lm:add(nn.SplitTable(1,2))
lm:add(nn.Sequencer(nn.Recurrent(6,nn.Identity(),nn.Linear(6,6)))) -- Add recurrent layer for input
lm:add(nn.Sequencer(nn.Linear(6,10))) -- Add activation function
lm:add(nn.Sequencer(nn.LogSoftMax())) -- Add Softmax layer in the end to determine next number.

-- Loss Function
criterion = nn.SequencerCriterion(nn.ClassNLLCriterion())

-- Training
for i=0,999 do
    local ip = torch.Tensor(trainset[inputs][i%10 + 1])
    local target = torch.Tensor(trainset[outputs][i%10 + 1])
    -- forward
    local op = lm:forward(ip)
    local err = criterion:forward(op, target)
    -- backward
    local gradOutputs = criterion:backward(op, target)
    lm:zeroGradParameters()
    lm:backward(ip, gradOutputs) -- Backward Through Time update
    -- update
    lm:updateParameters(0.1)
end

--Test
testip = torch.Tensor({1})
testop =lm:forward(testip)
pred,classes = testop[1]:sort(true)
print(classes[1])

testip = torch.Tensor({2})
testop =lm:forward(testip)
pred,classes = testop[1]:sort(true)
print(classes[1])

testip = torch.Tensor({3})
testop =lm:forward(testip)
pred,classes = testop[1]:sort(true)
print(classes[1])

testip = torch.Tensor({1,2,3,4})
testop =lm:forward(testip)
pred,classes = testop[4]:sort(true)
print(classes[1])
 
