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
  for c = 2, 5 do
    local p = torch.eq(prediction, c):double()
    local t = torch.eq(target, c):double()
    local true_positives = torch.mm(t:t(),p)
    true_positives = torch.sum(true_positives, 1)[1][1]
    local all_predicted = torch.sum(p, 1)[1][1]
    local all_targets = torch.sum(t, 1)[1][1]
    local precision = true_positives / all_predicted
    local recall = true_positives / all_targets
    
    local f1_score = 2 * precision * recall / (precision + recall)
    f1_accum = f1_accum + f1_score 
    precision_accum = precision_accum + precision
    recall_accum = recall_accum + recall
    
    
  end
  return {f1_accum / 4, precision_accum / 4, recall_accum / 4}
  
  
  
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

word_vectors = read_words('wordVectors.txt')
word_vectors = convert2tensors(word_vectors)

batch_size = 2
n_data = x_train:size(1)
data_index = 1

function gen_batch()
  end_index = data_index + batch_size
  if end_index > n_data then
    end_index = n_data
    data_index = 1
  end
  start_index = end_index - batch_size
  
  features = torch.Tensor(batch_size, x_train:size(2), word_vectors:size(2))
  labels = torch.Tensor(batch_size)
  for k = 1, batch_size do
    for i = 1, x_train:size(2) do 
      features[{{k}, {i}, {}}] = word_vectors[{{x_train[start_index + k - 1][i]}, {}}]
      labels[k] = y_train[start_index + k - 1][1]
    end
  end
  
  data_index = data_index + 1
  if data_index > n_data then 
    data_index = 1
  end
  return features, labels
end


x_raw = nn.Identity()()
x = nn.Reshape(x_train:size(2) * word_vectors:size(2))(x_raw)
h = nn.Linear(x_train:size(2) * word_vectors:size(2), 100)(x)
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

optim_state = {learningRate = 1e-2}

for i = 1, 1000 do
  local _, loss = optim.adagrad(feval, params, optim_state)
  if i % 10 == 0 then
    
    local features = x_dev[{{}, {}}]
    local labels = y_dev[{{}, 1}]
    local prediction, h = unpack(m:forward(features))
    local _, predicted_class  = prediction:max(2)
    local loss_dev = criterion:forward(prediction, labels)
    local f1_score, precision, recall = unpack(calc_f1(predicted_class, torch.reshape(labels, predicted_class:size(1), predicted_class:size(2))))
    print(string.format("loss_train = %6.8f, loss_dev = %6.8f, f1_score = %6.8f, precision = %6.8f, recall = %6.8f, gradnorm = %6.4e", loss[1], loss_dev, f1_score, precision, recall, grad_params:norm()))

  end
end



dummy_pass = 1




