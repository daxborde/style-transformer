import torch
import time
import os
from data import load_enron
from models import StyleTransformer, Discriminator
from train import train, auto_eval

def most_recent_path(rootpath, return_two=False):
    if rootpath is None:
        return None
    if not s.path.exists(rootpath):
        return None
    l = os.listdir(rootpath)
    if len(l) == 0:
        return None
    
    l.sort()
    ret = (os.path.join(rootpath, l[-1]), os.path.join(rootpath, l[-2]))
    if not return_two:
        ret = ret[0]
    return ret

class Config():
    data_path = './data/enronpa/'
    log_dir = 'runs/exp'
    save_path = './save'
    pretrained_embed_path = './embedding/'
    device = torch.device('cuda' if True and torch.cuda.is_available() else 'cpu')
    discriminator_method = 'Multi' # 'Multi' or 'Cond'
    load_pretrained_embed = False
    min_freq = 3
    max_length = 32
    embed_size = 256
    d_model = 256
    h = 4
    num_styles = 2
    num_classes = num_styles + 1 if discriminator_method == 'Multi' else 2
    num_layers = 4
    batch_size = 64
    lr_F = 0.0001
    lr_D = 0.0001
    L2 = 0
    iter_D = 10
    iter_F = 5
    F_pretrain_iter = 500
    log_steps = 5
    eval_steps = 25
    learned_pos_embed = True
    dropout = 0
    drop_rate_config = [(1, 0)]
    temperature_config = [(1, 0)]

    slf_factor = 0.25
    cyc_factor = 0.5
    adv_factor = 1

    inp_shuffle_len = 0
    inp_unk_drop_fac = 0
    inp_rand_drop_fac = 0
    inp_drop_prob = 0

    loss_log = './save/loss_log.txt'

def main():
    config = Config()
    train_iters, test_iters, vocab = load_enron(config)
    print('Vocab size:', len(vocab))
    model_F = StyleTransformer(config, vocab).to(config.device)
    model_D = Discriminator(config, vocab).to(config.device)
    print(config.discriminator_method)

    # last_checkpoint = most_recent_path(most_recent_path(config.save_path), return_two=True)
    # if last_checkpoint:
    #     print(last_checkpoint)
    #     model_D.load_state_dict(torch.load(last_checkpoint[1]))
    #     model_F.load_state_dict(torch.load(last_checkpoint[0]))

    train(config, vocab, model_F, model_D, train_iters, test_iters)
    
if __name__ == '__main__':
    main()