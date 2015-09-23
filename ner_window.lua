require 'mobdebug'.start()
require 'nn'
require 'nngraph'
require 'optim'
require 'Embedding'
local model_utils=require 'model_utils'
require 'table_utils'
nngraph.setDebug(true)


function calc_f1(prediction, target)
  local f1_accum = 0
  local precision_accum = 0
  local recall_accum = 0
  for c = 1, 5 do
    local p = torch.eq(prediction, c):double()
    local t = torch.eq(target, c):double()
    local true_positives = torch.mm(t:t(),p)[1][1]
        
    p = torch.eq(prediction, c):double()
    t = torch.ne(target, c):double()
    local false_positives = torch.mm(t:t(),p)[1][1]
    
    p = torch.ne(prediction, c):double()
    t = torch.eq(target, c):double()
    local false_negatives = torch.mm(t:t(),p)[1][1]
    
    local precision = true_positives / (true_positives + false_positives)
    local recall = true_positives / (true_positives + false_negatives)
    
    local f1_score = 2 * precision * recall / (precision + recall)
    f1_accum = f1_accum + f1_score 
    precision_accum = precision_accum + precision
    recall_accum = recall_accum + recall
    
    
  end
  return {f1_accum / 5, precision_accum / 5, recall_accum / 5}
  
  
  
  
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
x = nn.Reshape(batch_size * x_train:size(2))(x_raw)
x = Embedding(vocab_size, 50)(x)
x = nn.Reshape(batch_size, 50 * x_train:size(2))(x)
h = nn.Linear(50 * x_train:size(2), 100)(x)
h = nn.Tanh()(h)
z = nn.Linear(100, 5)(h)
z = nn.SoftMax()(z)
m = nn.gModule({x_raw}, {z, h})


local params, grad_params = model_utils.combine_all_parameters(m)
params:uniform(-0.08, 0.08)

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

optim_state = {learningRate = 1e-2}

for i = 1, 200 do
  local _, loss = optim.adam(feval, params, optim_state)
  if i % 10 == 0 then
    
    local loss_train = loss[1]
    local _, predicted_class  = prediction:max(2)
    local f1_score_train, precision_train, recall_train = unpack(calc_f1(predicted_class, torch.reshape(labels, predicted_class:size(1), predicted_class:size(2))))
        
    local features = x_dev[{{1, batch_size}, {}}]
    local labels = y_dev[{{1, batch_size}, 1}]
    local prediction, h = unpack(m:forward(features))
    local _, predicted_class  = prediction:max(2)
    local loss_dev = criterion:forward(prediction, labels)
    local f1_score_dev, precision_dev, recall_dev = unpack(calc_f1(predicted_class, torch.reshape(labels, predicted_class:size(1), predicted_class:size(2))))
    print(string.format("train set: loss = %6.8f, f1_score = %6.8f, precision = %6.8f, recall = %6.8f, grad_params:norm() = %6.4e", loss_train, f1_score_train, precision_train, recall_train, grad_params:norm()))
    print(string.format("dev set:   loss = %6.8f, f1_score = %6.8f, precision = %6.8f, recall = %6.8f", loss_dev, f1_score_dev, precision_dev, recall_dev))

  end
end


-- input words that give the highest probability P(class | word)

for considered_class = 1, 5 do
  current_argmax = 1
  current_max = -1e+12
  for k = 1, math.floor(vocab_size / batch_size) do 
    
    features = torch.ones(batch_size, 3)
    for i = 1 + batch_size*(k-1), batch_size * k - 1 do 
      features[i%batch_size][2] = i
    end
    local prediction, h = unpack(m:forward(features))
    local predicted_max, predicted_argmax = prediction[{{}, {considered_class}}]:max(1)
    predicted_max = predicted_max[1][1]
    predicted_argmax = predicted_argmax[1][1]
    if predicted_max > current_max then
      current_max = predicted_max
      current_argmax = predicted_argmax
    end
      
  end
  print(string.format('%s has highest probability for class %d', vocabulary[current_argmax], considered_class)
end

dummy_pass = 1





