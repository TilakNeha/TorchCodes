require 'nn'
require 'rnn'
require 'math'
CharSplitLMMinibatchLoader = require 'CharSplitLMMinibatchLoader'
local ok, cutorch = pcall(require, 'cutorch')
if not ok then print('package cutorch not found!') end
if ok then
    print('using CUDA on GPU ')
else
    print('Falling back on CPU mode')
end

data_dir = 'data'
batch_size = 5
rho = 5 -- number of timesteps to unroll for
split_sizes = {1,0,0} -- train, val, test
loader = CharSplitLMMinibatchLoader.create(data_dir, batch_size, rho, split_sizes)
itov = {}
vocab_size = 0
for k,v in pairs(loader.vocab_mapping) do 
    itov[v] = k 
    vocab_size = vocab_size + 1
end

rnn = nn.Sequential()
rnn:add(nn.LookupTable(vocab_size,30))
rnn:add(nn.SplitTable(1,2))
rnn:add(nn.Sequencer(nn.Recurrent(30,nn.Identity(),nn.Linear(30,30))))
rnn:add(nn.Sequencer(nn.Linear(30,vocab_size)))
rnn:add(nn.Sequencer(nn.Tanh()))
rnn:add(nn.Sequencer(nn.LogSoftMax()))

criterion = nn.ModuleCriterion(
            nn.SequencerCriterion(nn.ClassNLLCriterion()),
            nil,
            nn.SplitTable(1,1):type('torch.IntTensor')
)

lm = nn.Sequential()
lm:add(nn.LookupTable(vocab_size,vocab_size))
lm:add(nn.SplitTable(1,2))
local encLSTM = nn.LSTM(vocab_size, vocab_size)
local encLSTM2 = nn.LSTM(vocab_size, vocab_size)
lm:add(nn.Sequencer(encLSTM))
lm:add(nn.Sequencer(nn.Linear(vocab_size,vocab_size)))
lm:add(nn.Sequencer(nn.Tanh()))
lm:add(nn.Sequencer(encLSTM2))
lm:add(nn.Sequencer(nn.Linear(vocab_size,vocab_size)))
lm:add(nn.Sequencer(nn.Tanh()))
lm:add(nn.Sequencer(nn.Linear(vocab_size,vocab_size)))
lm:add(nn.Sequencer(nn.LogSoftMax()))

if ok then
   rnn = rnn:cuda()
   criterion = criterion:cuda()
   lm = lm:cuda()
end

-- Training
local batch
num_iterations = 1000 
for i=1,num_iterations do
    local inputs, targets = loader:next_batch(1)
    if ok then
       inputs = inputs:cuda()
       target = targets:cuda()
    end
    -- forward
    local outputs = lm:forward(inputs)
    local err = criterion:forward(outputs,targets)
    -- backward
    local gradOutputs = criterion:backward(outputs,targets)
    lm:zeroGradParameters()
    lm:backward(inputs,gradOutputs)
    -- update
    lm:updateParameters(0.1)
    print('Iteration: '..i..'/'..num_iterations.. ' Error: '.. err)
end

-- Sampling
start = 'a'
sampling_len = 200
sampled_str = ""
sampled_str = sampled_str..start
for i=1,sampling_len do
    pred = lm:forward(torch.Tensor({loader.vocab_mapping[start]}))
    _,ind = pred[1]:sort()
    s = ind:size()[1]
    rand = math.random(s,s)
    start = itov[ind[rand]]
    sampled_str = sampled_str..start
end
print(sampled_str)    

--lm:evaluate()
--start ='a'
--sampling_length = 1000
--tempreture = 1
--sampled_str = ""
--for i=1, sampling_length do
--    next_word_dist = rnn:forward(torch.Tensor({loader.vocab_mapping[start]}))[1]
--    next_word_dist:div(tempreture) -- scale by temperature
--    local probs = torch.exp(next_word_dist):squeeze()
--    probs:div(torch.sum(probs)) -- renormalize so probs sum to one
--    current_word = torch.multinomial(probs:float(), 1):resize(1)
--    --_,ind = probs:float():sort()
--    --indx = ind:size()[1]
--    --current_word = ind[indx]
--    sampled_str = sampled_str..itov[current_word[1]]
--    start = itov[current_word[1]]
--end
--print(sampled_str)

