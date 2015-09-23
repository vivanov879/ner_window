require 'mobdebug'.start()
require 'nn'
require 'nngraph'
require 'optim'
require 'Embedding'
local model_utils=require 'model_utils'
require 'table_utils'
nngraph.setDebug(true)


embed = Embedding(2, 3)
params, grad_params = embed:getParameters()

embed_clones = model_utils.clone_many_times(embed, 2)

params1, grad_params1 = model_utils.combine_all_parameters(embed_clones[1])
params2, grad_params2 = model_utils.combine_all_parameters(embed_clones[2])

params11, grad_params11 = embed_clones[1]:getParameters()
params22, grad_params22 = embed_clones[2]:getParameters()

params1 = params
params2 = params

params[1] = 1

print(params[1])
print(params1[1])
print(params2[1])
print(params11[1])
print(params22[1])
dummy_pass = 1
