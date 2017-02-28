exp_name = 'darknet19_voc07trainval_exp1'

pretrained_fname = 'darknet19.weights.npz'

start_step = 0
end_step = 100000
lr_decay_steps = {60000, 80000}
lr_decay = 1./10