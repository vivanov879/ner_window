require 'mobdebug'.start()
require 'nn'
require 'nngraph'
require 'optim'
require 'Embedding'
local model_utils=require 'model_utils'
require 'table_utils'
nngraph.setDebug(true)


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

batch_size = 2
n_data = x_train:size(1)

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

dummy_pass = 1

