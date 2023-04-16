import os
from fewshot_re_kit.data_loader import get_loader, get_rel2id
from fewshot_re_kit.framework import FewShotREFramework
from fewshot_re_kit.sentence_encoder import BERTSentenceEncoder
from models.CBPM import CBPM
import numpy as np
import argparse
import torch
import random
import time
import datetime


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='./data',
                        help='file root')
    parser.add_argument('--train', default='train',
                        help='train file')
    parser.add_argument('--val', default='val',
                        help='val file')
    parser.add_argument('--test', default='test_wiki',
                        help='test file')
    parser.add_argument('--ispubmed', default=False, type=bool,
                        help='FewRel 2.0 or not')
    parser.add_argument('--pid2name', default='pid2name',
                        help='pid2name file: relation names and description')
    parser.add_argument('--trainN', default=5, type=int,
                        help='N in train')
    parser.add_argument('--N', default=5, type=int,
                        help='N way')
    parser.add_argument('--K', default=1, type=int,
                        help='K shot')
    parser.add_argument('--Q', default=10, type=int,
                        help='Num of query per class')
    parser.add_argument('--batch_size', default=1, type=int,
                        help='batch size')
    parser.add_argument('--train_iter', default=10000, type=int,
                        help='num of iters in training')
    parser.add_argument('--val_iter', default=1000, type=int,
                        help='num of iters in validation')
    parser.add_argument('--test_iter', default=10000, type=int,
                        help='num of iters in testing')
    parser.add_argument('--val_step', default=1000, type=int,
                        help='val after training how many iters')
    parser.add_argument('--model', default='Enhance', choices=['Enhance'],
                        help='model name')
    parser.add_argument('--encoder', default='bert',
                        help='encoder: bert')
    parser.add_argument('--max_length', default=128, type=int,
                        help='max length')
    parser.add_argument('--lr', default=2e-5, type=float,
                        help='learning rate')
    parser.add_argument('--weight_decay', default=0.01, type=float,
                        help='weight decay')
    parser.add_argument('--grad_iter', default=1, type=int,
                        help='accumulate gradient every x iterations')
    parser.add_argument('--optim', default='adamw',
                        help='sgd / adam / adamw')
    parser.add_argument('--hidden_size', default=768, type=int,
                        help='hidden size')
    parser.add_argument('--load_ckpt', default=None,
                        help='load ckpt')
    parser.add_argument('--save_ckpt', default=None,
                        help='save ckpt')
    parser.add_argument('--only_test', default=False,
                        help='only test')
    parser.add_argument('--pretrain_ckpt', default='bert-base-uncased',
                        help='bert / roberta pre-trained checkpoint')
    parser.add_argument('--seed', default=2020, type=int,
                        help='seed')
    parser.add_argument('--path', default=None,
                        help='path to ckpt')
    parser.add_argument('--hierarchical', default='data/sim.pkl',
                        help='hierarchical file path')
    parser.add_argument('--sentence_encoder', default='bert', choices=['bert'])
    parser.add_argument('--mlp_hidden_state', default=1536 * 2, type=int, help='')
    opt = parser.parse_args()
    if opt.seed:
        random.seed(opt.seed)
        np.random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed_all(opt.seed)
        print(f'seed: {opt.seed}')
    print(opt)
    print(random.randint(1, 10000))


    print("{}-way-{}-shot Few-Shot Relation Classification".format(opt.N, opt.K))
    print("model: {}".format(opt.model))
    print("encoder: {}".format(opt.encoder))
    print("max_length: {}".format(opt.max_length))

    # encoder
    sentence_encoder = BERTSentenceEncoder(opt.pretrain_ckpt, opt.max_length, path=opt.path)

    # get id2rel and rel2id
    rel2id, id2rel = get_rel2id(root=opt.root, paths=[opt.train, opt.val, opt.test])

    # train / val / test data loader
    train_data_loader = get_loader(opt.train, opt.pid2name, sentence_encoder,
                                   N=opt.trainN, K=opt.K, Q=opt.Q, batch_size=opt.batch_size, ispubmed=opt.ispubmed, root=opt.root,
                                   rel2id=rel2id, id2rel=id2rel, seed=opt.seed)
    val_data_loader = get_loader(opt.val, opt.pid2name, sentence_encoder,
                                 N=opt.N, K=opt.K, Q=opt.Q, batch_size=opt.batch_size, ispubmed=opt.ispubmed, root=opt.root,
                                 rel2id=rel2id, id2rel=id2rel, seed=opt.seed)
    test_data_loader = get_loader(opt.test, opt.pid2name, sentence_encoder,
                                  N=opt.N, K=opt.K, Q=opt.Q, batch_size=opt.batch_size, ispubmed=opt.ispubmed, root=opt.root,
                                  rel2id=rel2id, id2rel=id2rel, seed=opt.seed)

    framework = FewShotREFramework(train_data_loader, val_data_loader, test_data_loader, opt)
    model = CBPM(sentence_encoder, hidden_size=opt.hidden_size, max_len=opt.max_length,
                 hierarchical=opt.hierarchical, rel2id=rel2id, id2rel=id2rel, opt=opt)
    if torch.cuda.is_available():
        model.cuda()
    print(f'we use model {opt.model}')

    s_p = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    if not os.path.exists('checkpoint'):
        os.mkdir('checkpoint')
    os.mkdir('checkpoint/' + s_p)
    hie = 'hierarchical'
    prefix = '-'.join([opt.model, opt.encoder, opt.train, opt.val, str(opt.N), str(opt.K), hie])
    ckpt = 'checkpoint/{}/{}.pth.tar'.format(s_p, prefix)
    print(f'model saved as {ckpt}')
    if opt.save_ckpt:
        ckpt = opt.save_ckpt

    if not opt.only_test:
        start_time = time.perf_counter()
        framework.train(model, prefix, opt.trainN, opt.N, opt.K, opt.Q, learning_rate=opt.lr, weight_decay=opt.weight_decay,
                        train_iter=opt.train_iter, val_iter=opt.val_iter,
                        load_ckpt=opt.load_ckpt, save_ckpt=ckpt, val_step=opt.val_step, grad_iter=opt.grad_iter,
                        opt=opt)
        end_time = time.perf_counter()
        print('total training time:%s s' % (end_time - start_time))
    else:
        ckpt = opt.load_ckpt

    start_time = time.perf_counter()
    acc = framework.eval(model, opt.N, opt.K, opt.Q, opt.test_iter, ckpt=ckpt)
    end_time = time.perf_counter()
    print(f'model saved as {ckpt}')
    print('total evaluation time:%s s' % (end_time - start_time))
    print("RESULT: %.2f" % (acc * 100))


if __name__ == "__main__":
    main()
