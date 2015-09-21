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

  sentences = sentences_en
  
  local batch = torch.Tensor(batch_size, 3)
  local target = 1
  if data_index % 2 == 0 then
    target = -1
  end
  for k = 1, batch_size do
    sentence = sentences[start_index + k - 1]
    center_word_index = math.random(#sentence)
    center_word = sentence[center_word_index]
    context_index = center_word_index + (math.random() > 0.5 and 1 or -1) * math.random(2, math.floor(context_size/2))
    context_index = math.clamp(context_index, 1, #sentence)
    outer_word = sentence[context_index]
    neg_word = math.random(#vocabulary_en)
    batch[k][1] = center_word
    if target == 1 then
      batch[k][2] = outer_word
    else 
      batch[k][2] = neg_word
    end
  end
  data_index = data_index + 1
  if data_index > n_data then 
    data_index = 1
  end
  return batch, target
end



