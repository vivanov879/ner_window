require 'mobdebug'.start()
require 'nn'
require 'nngraph'
require 'optim'
require 'Embedding'
local model_utils=require 'model_utils'
require 'table_utils'
nngraph.setDebug(true)


function gen_data()
  --заделаем тестовые features и labels
  local n_data = 100
  local n_features = 3
  local n_labels = 2
  local features_input = torch.zeros(n_data, n_features)
  local labels_input = torch.zeros(n_data, n_labels)
  for i = 1, n_data do 
    element = torch.ones(n_features)

    element:mul(i)
    element[2] = 0.5
    element[3] = 1
    element:add(torch.randn(element:size()))
    element:div(n_data)
    features_input[{{i}, {}}] = element
    
    label = torch.ones(n_labels)
    label[2] =  i
    label[1] = i ^ 2
    label:add(torch.randn(label:size()))
    label:div(n_data ^ 2)
    labels_input[{{i}, {}}] = label
    
  end
  return features_input, labels_input
end

features1, labels = gen_data()
features2, labels = gen_data()

source = nn.Linear(3,2)

local params1, grad_params1 = model_utils.combine_all_parameters(source)
params1:uniform(-0.08, 0.08)


clones = model_utils.clone_many_times(source, 2)


x1 = nn.Identity()()
x2 = nn.Identity()()

z1 = nn.Linear(3,2)(x1) 
z2 = nn.Linear(3,2)(x2) 

z = nn.CAddTable()({z1, z2})
m = nn.gModule({x1, x2}, {z})

local params, grad_params = model_utils.combine_all_parameters(m)
params:uniform(-0.08, 0.08)

criterion = nn.MSECriterion()

function feval(x_arg)
    if x_arg ~= params then
        params:copy(x_arg)
    end
    grad_params:zero()
    
    local loss = 0
    
    ------------------- forward pass -------------------
    prediction = m:forward({features1, features2})
    loss_m = criterion:forward(prediction, labels)
    loss = loss + loss_m
    
    -- complete reverse order of the above
    dprediction = criterion:backward(prediction, labels)
    dfeatures1, dfeatures2 = unpack(m:backward({features1, features2}, dprediction))
    
    return loss, grad_params

end

optim_state = {learningRate = 1e-3}


for i = 1, 10000 do
  local _, loss = optim.adagrad(feval, params, optim_state)
  if i % 10 == 0 then
      print(string.format("iteration %4d, loss = %6.6f", i, loss[1]))
      --print(params)
      
  end
end
