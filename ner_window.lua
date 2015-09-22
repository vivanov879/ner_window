require 'mobdebug'.start()
require 'nn'
require 'nngraph'
require 'optim'
require 'Embedding'
local model_utils=require 'model_utils'
require 'table_utils'
nngraph.setDebug(true)


function calc_f1(prediction, target)
  local accum = 0
  for c = 2, 5 do
    local p = prediction:clone()
    local t = target:clone()
    p = torch.eq(p, c):double()
    t = torch.eq(t, c):double()
    d = torch.add(p, -t)
    d = torch.abs(d)
    local true_positives = torch.sum(d, 1)[1][1]
    p = prediction:clone()
    p = torch.eq(p, c):double()
    local all_predicted = torch.sum(p, 1)[1][1]
    t = target:clone()
    local all_targets = torch.sum(t, 1)[1][1]
    local precision = true_positives / all_predicted
    local recall = true_positives / all_targets
    
    local f1_score = 2 * precision * recall / (precision + recall)
    accum = accum + f1_score 
    
  end
  return accum / 4
  
  
  
end

function read_words(fn)
  fd = io.lines(fn)
  sentences = {}
  line = fd()

  while line do
    sentence = {}
    for _, word in pairs(string.split(line, " ")) do
        sentence[#sentence + 1] = word
    end
    sentences[#sentences + 1] = sentence
    line = fd()
  end
  return sentences
end


vocabulary_raw = read_words('vocabulary_raw')
inv_vocabulary_raw = read_words('inv_vocabulary_raw')
x_train_raw = read_words('x_train')
y_train_raw = read_words('y_train')

vocabulary = {}
inv_vocabulary = {}

for i, sentence in pairs(vocabulary_raw) do 
  vocabulary[tonumber(sentence[1])] = sentence[2]
  inv_vocabulary[sentence[2]] = tonumber(sentence[1])
end

vocab_size = #vocabulary

function convert2tensors(sentences)
  local t = torch.Tensor(#sentences, #sentences[1])
  for k, sentence in pairs(sentences) do
    assert(#sentence == #sentences[1])
    for i = 1, #sentence do 
      t[k][i] = tonumber(sentence[i])
    end
  end
  return t  
end

x_train = convert2tensors(x_train_raw)
y_train = convert2tensors(y_train_raw)

x_dev_raw = read_words('x_dev')
y_dev_raw = read_words('y_dev')
x_dev = convert2tensors(x_dev_raw)
y_dev = convert2tensors(y_dev_raw)


batch_size = 10000
n_data = x_train:size(1)
data_index = 1

function gen_batch()
  end_index = data_index + batch_size
  if end_index > n_data then
    end_index = n_data
    data_index = 1
  end
  start_index = end_index - batch_size
  
  features = x_train[{{data_index, data_index + batch_size - 1}, {}}]
  labels = y_train[{{data_index, data_index + batch_size - 1}, 1}]
    
  data_index = data_index + 1
  if data_index > n_data then 
    data_index = 1
  end
  return features, labels
end


x_raw = nn.Identity()()
l = {}
for i = 1, x_train:size(2) do
  x = nn.Select(2,i)(x_raw)
  x = Embedding(vocab_size, 50)(x)
  l[#l + 1] = x
end
x = nn.JoinTable(2)(l)
h = nn.Linear(50 * x_train:size(2), 100)(x)
h = nn.Tanh()(h)
z = nn.Linear(100, 5)(h)
z = nn.SoftMax()(z)
m = nn.gModule({x_raw}, {z, h})


local params, grad_params = model_utils.combine_all_parameters(m)
params:uniform(-0.15, 0.15)

criterion = nn.ClassNLLCriterion()



function feval(x_arg)
    if x_arg ~= params then
        params:copy(x_arg)
    end
    grad_params:zero()
    
    local loss = 0
    
    features, labels = gen_batch()
            
    ------------------- forward pass -------------------
    prediction, h = unpack(m:forward(features))
    loss_m = criterion:forward(prediction, labels)
    loss = loss + loss_m
    
    -- complete reverse order of the above
    dh = torch.zeros(h:size())
    dprediction = criterion:backward(prediction, labels)
    dfeatures = m:backward(features, {dprediction, dh})
    
    return loss, grad_params

end

optim_state = {learningRate = 1e-2, weightDecay = 1e-6}

for i = 1, 1000 do
  local _, loss = optim.sgd(feval, params, optim_state)
  if i % 10 == 0 then
    local _, predicted_class  = prediction:max(2)
    print(prediction[{1, {}}])
    print(string.format("center_word = %s, predicted class = %d, target class = %d, loss = %6.8f, gradnorm = %6.4e", vocabulary[features[1][math.floor(features:size(2) / 2)]], predicted_class[1][1], labels[1], loss[1], grad_params:norm()))
  end
  if i % 10 == 0 then 
    features = x_dev[{{}, {}}]
    labels = y_dev[{{}, 1}]
    prediction, h = unpack(m:forward(features))
    loss = criterion:forward(prediction, labels)
    
    local _, predicted_class  = prediction:max(2)
    
    f1_score = calc_f1(predicted_class, torch.reshape(labels, predicted_class:size(1), predicted_class:size(2)))
    print(string.format('dev loss = %6.8f, f1_score = %6.8f', loss, f1_score))

  end
end



dummy_pass = 1





