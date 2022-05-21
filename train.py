from __future__ import print_function
import argparse
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from data_loader import SYSUData, RegDBData, TestData
from data_manager import *
from eval_metrics import eval_sysu, eval_regdb
from model import embed_net
from utils import *
from loss import OriTripletLoss, TriLoss, DCLoss
from tensorboardX import SummaryWriter
from random_erasing import RandomErasing

parser = argparse.ArgumentParser(description='PyTorch Cross-Modality Training')
parser.add_argument('--dataset', default='sysu', help='dataset name: regdb or sysu]')
parser.add_argument('--lr', default=0.1 , type=float, help='learning rate, 0.00035 for adam')
parser.add_argument('--optim', default='sgd', type=str, help='optimizer')
parser.add_argument('--arch', default='resnet50', type=str, help='network baseline:resnet18 or resnet50')
parser.add_argument('--resume', '-r', default='', type=str, help='resume from checkpoint')
parser.add_argument('--test-only', action='store_true', help='test only')
parser.add_argument('--model_path', default='save_model/', type=str, help='model save path')
parser.add_argument('--save_epoch', default=20, type=int, metavar='s', help='save model every 10 epochs')
parser.add_argument('--log_path', default='log/', type=str, help='log save path')
parser.add_argument('--vis_log_path', default='log/vis_log/', type=str, help='log save path')
parser.add_argument('--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('--img_w', default=192, type=int, metavar='imgw', help='img width')
parser.add_argument('--img_h', default=384, type=int, metavar='imgh', help='img height')
parser.add_argument('--batch-size', default=4, type=int, metavar='B', help='training batch size')
parser.add_argument('--test-batch', default=64, type=int, metavar='tb', help='testing batch size')
parser.add_argument('--method', default='agw', type=str, metavar='m', help='method type: base or agw')
parser.add_argument('--margin', default=0.3, type=float, metavar='margin', help='triplet loss margin')
parser.add_argument('--num_pos', default=4, type=int, help='num of pos per identity in each modality')
parser.add_argument('--trial', default=1, type=int, metavar='t', help='trial (only for RegDB dataset)')
parser.add_argument('--seed', default=0, type=int, metavar='t', help='random seed')
parser.add_argument('--gpu', default='0', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')
parser.add_argument('--mode', default='all', type=str, help='all or indoor')
parser.add_argument('--delta', default=0.2, type=float, metavar='delta', help='dcl weights, 0.2 for PCB, 0.5 for resnet50')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

set_seed(args.seed)

dataset = args.dataset
if dataset == 'sysu':
    data_path = './Datasets/SYSU-MM01/'
    log_path = args.log_path + 'sysu_log/'
    test_mode = [1, 2]  # thermal to visible
elif dataset == 'regdb':
    data_path = './Datasets/RegDB/'
    log_path = args.log_path + 'regdb_log/'
    test_mode = [2, 1]  # visible to thermal

checkpoint_path = args.model_path

if not os.path.isdir(log_path):
    os.makedirs(log_path)
if not os.path.isdir(checkpoint_path):
    os.makedirs(checkpoint_path)
if not os.path.isdir(args.vis_log_path):
    os.makedirs(args.vis_log_path)

suffix = dataset
if args.method=='agw':
    suffix = suffix + '_agw_p{}_n{}_lr_{}_seed_{}'.format(args.num_pos, args.batch_size, args.lr, args.seed)
else:
    suffix = suffix + '_base_p{}_n{}_lr_{}_seed_{}'.format(args.num_pos, args.batch_size, args.lr, args.seed)


if not args.optim == 'sgd':
    suffix = suffix + '_' + args.optim

if dataset == 'regdb':
    suffix = suffix + '_trial_{}'.format(args.trial)

sys.stdout = Logger(log_path + suffix + '_os.txt')

vis_log_dir = args.vis_log_path + suffix + '/'

if not os.path.isdir(vis_log_dir):
    os.makedirs(vis_log_dir)
writer = SummaryWriter(vis_log_dir)
print("==========\nArgs:{}\n==========".format(args))
device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0

print('==> Loading data..')
# Data loading code
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform_train = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Pad(10),
    transforms.RandomCrop((args.img_h, args.img_w)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
    RandomErasing(probability = 0.5, mean=[0.0, 0.0, 0.0]),
])
transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((args.img_h, args.img_w)),
    transforms.ToTensor(),
    normalize,
])

end = time.time()
if dataset == 'sysu':
    # training set
    trainset = SYSUData(data_path, transform=transform_train)
    # generate the idx of each person identity
    color_pos, thermal_pos = GenIdx(trainset.train_color_label, trainset.train_thermal_label)

    # testing set
    query_img, query_label, query_cam = process_query_sysu(data_path, mode=args.mode)
    gall_img, gall_label, gall_cam = process_gallery_sysu(data_path, mode=args.mode, trial=0)

elif dataset == 'regdb':
    # training set
    trainset = RegDBData(data_path, args.trial, transform=transform_train)
    # generate the idx of each person identity
    color_pos, thermal_pos = GenIdx(trainset.train_color_label, trainset.train_thermal_label)

    # testing set
    query_img, query_label = process_test_regdb(data_path, trial=args.trial, modal='visible')
    gall_img, gall_label = process_test_regdb(data_path, trial=args.trial, modal='thermal')

gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(args.img_w, args.img_h))
queryset = TestData(query_img, query_label, transform=transform_test, img_size=(args.img_w, args.img_h))

# testing data loader
gall_loader = data.DataLoader(gallset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

n_class = len(np.unique(trainset.train_color_label))
nquery = len(query_label)
ngall = len(gall_label)

print('Dataset {} statistics:'.format(dataset))
print('  ------------------------------')
print('  subset   | # ids | # images')
print('  ------------------------------')
print('  visible  | {:5d} | {:8d}'.format(n_class, len(trainset.train_color_label)))
print('  thermal  | {:5d} | {:8d}'.format(n_class, len(trainset.train_thermal_label)))
print('  ------------------------------')
print('  query    | {:5d} | {:8d}'.format(len(np.unique(query_label)), nquery))
print('  gallery  | {:5d} | {:8d}'.format(len(np.unique(gall_label)), ngall))
print('  ------------------------------')
print('Data Loading Time:\t {:.3f}'.format(time.time() - end))

print('==> Building model..')
if args.method =='base':
    net = embed_net(n_class, no_local= 'off', gm_pool =  'off', arch=args.arch)
else:
    net = embed_net(n_class, no_local= 'on', gm_pool = 'on', arch=args.arch)
net.to(device)
cudnn.benchmark = True

if len(args.resume) > 0:
    model_path = checkpoint_path + args.resume
    if os.path.isfile(model_path):
        print('==> loading checkpoint {}'.format(args.resume))
        checkpoint = torch.load(model_path)
        start_epoch = checkpoint['epoch']
        net.load_state_dict(checkpoint['net'])
        print('==> loaded checkpoint {} (epoch {})'
              .format(args.resume, checkpoint['epoch']))
    else:
        print('==> no checkpoint found at {}'.format(args.resume))

# define loss function
criterion_id = nn.CrossEntropyLoss()

loader_batch = args.batch_size * args.num_pos
criterion_tri= OriTripletLoss(batch_size=loader_batch, margin=args.margin)
self_critial= TriLoss(batch_size=loader_batch, margin=args.margin)
criterion_div = DCLoss(num=2)

criterion_id.to(device)
criterion_tri.to(device)
criterion_div.to(device)


if args.optim == 'sgd':
    ignored_params = list(map(id, net.bottleneck1.parameters())) \
                     + list(map(id, net.bottleneck2.parameters())) \
                     + list(map(id, net.bottleneck3.parameters())) \
                     + list(map(id, net.bottleneck4.parameters())) \
                     + list(map(id, net.classifier1.parameters())) \
                     + list(map(id, net.classifier2.parameters())) \
                     + list(map(id, net.classifier3.parameters())) \
                     + list(map(id, net.classifier4.parameters()))

    base_params = filter(lambda p: id(p) not in ignored_params, net.parameters())

    optimizer = optim.SGD([
        {'params': base_params, 'lr': 0.1 * args.lr},
        {'params': net.bottleneck1.parameters(), 'lr': args.lr},
        {'params': net.bottleneck2.parameters(), 'lr': args.lr},
        {'params': net.bottleneck3.parameters(), 'lr': args.lr},
        {'params': net.bottleneck4.parameters(), 'lr': args.lr},
        {'params': net.classifier1.parameters(), 'lr': args.lr},
        {'params': net.classifier2.parameters(), 'lr': args.lr},
        {'params': net.classifier3.parameters(), 'lr': args.lr},
        {'params': net.classifier4.parameters(), 'lr': args.lr}],
        weight_decay=5e-4, momentum=0.9, nesterov=True)

# exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if epoch < 10:
        lr = args.lr * (epoch + 1) / 10
    elif epoch >= 10 and epoch < 20:
        lr = args.lr
    elif epoch >= 20 and epoch < 50:
        lr = args.lr * 0.1
    elif epoch >= 50:
        lr = args.lr * 0.01

    optimizer.param_groups[0]['lr'] = 0.1 * lr
    for i in range(len(optimizer.param_groups) - 1):
        optimizer.param_groups[i + 1]['lr'] = lr

    return lr

def train(epoch):

    current_lr = adjust_learning_rate(optimizer, epoch)
    train_loss = AverageMeter()
    id_loss = AverageMeter()
    tri_loss = AverageMeter()
    data_time = AverageMeter()
    batch_time = AverageMeter()
    correct = 0
    total = 0

    # switch to train mode
    net.train()
    end = time.time()

    for batch_idx, (input1, input2, label1, label2) in enumerate(trainloader):

        labels = torch.cat((label1, label1, label2, label2), 0)

        input1 = Variable(input1.cuda())
        input2 = Variable(input2.cuda())

        labels = Variable(labels.cuda())
        data_time.update(time.time() - end)


        feat1, feat2, feat3, feat4, out1, out2, out3, out4, g1 = net(input1, input2)

        loss_id = (criterion_id(out1, labels) + criterion_id(out2, labels) + criterion_id(out3, labels) + criterion_id(out4, labels))*0.25
        
        lbs = torch.cat((label1, label2), 0)

        ft11, ft12, ft13, ft14 = torch.chunk(feat1, 4, 0)
        ft21, ft22, ft23, ft24 = torch.chunk(feat2, 4, 0)
        ft31, ft32, ft33, ft34 = torch.chunk(feat3, 4, 0)
        ft41, ft42, ft43, ft44 = torch.chunk(feat4, 4, 0)
        ba= criterion_tri(ft11, label1)[1]

        loss_tri1 = (criterion_tri(torch.cat((ft11, ft13),0), lbs)[0] + criterion_tri(torch.cat((ft11, ft14),0), lbs)[0] + criterion_tri(torch.cat((ft12, ft13),0), lbs)[0] + criterion_tri(torch.cat((ft12, ft14),0), lbs)[0])/4
        loss_tri2 = (criterion_tri(torch.cat((ft21, ft23),0), lbs)[0] + criterion_tri(torch.cat((ft21, ft24),0), lbs)[0] + criterion_tri(torch.cat((ft22, ft23),0), lbs)[0] + criterion_tri(torch.cat((ft22, ft24),0), lbs)[0])/4
        loss_tri3 = (criterion_tri(torch.cat((ft31, ft33),0), lbs)[0] + criterion_tri(torch.cat((ft31, ft34),0), lbs)[0] + criterion_tri(torch.cat((ft32, ft33),0), lbs)[0] + criterion_tri(torch.cat((ft32, ft34),0), lbs)[0])/4
        loss_tri4 = (criterion_tri(torch.cat((ft41, ft43),0), lbs)[0] + criterion_tri(torch.cat((ft41, ft44),0), lbs)[0] + criterion_tri(torch.cat((ft42, ft43),0), lbs)[0] + criterion_tri(torch.cat((ft42, ft44),0), lbs)[0])/4
        
        loss_tri = (loss_tri1 + loss_tri2 + loss_tri3 + loss_tri4)/4
        
        loss_dcl = (criterion_div(torch.cat((ft12, ft14),0)) + criterion_div(torch.cat((ft22, ft24),0)) + criterion_div(torch.cat((ft32, ft34),0)) + criterion_div(torch.cat((ft42, ft44),0)))*0.25*args.delta


        correct += (ba / 2)
        _, predicted = out1.max(1)
        correct += (predicted.eq(labels).sum().item() / 2)

        loss = loss_id + loss_tri + loss_dcl
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update P
        train_loss.update(loss.item(), 2 * input1.size(0))
        id_loss.update(loss_id.item(), 2 * input1.size(0))
        tri_loss.update(loss_tri.item(), 2 * input1.size(0))
        total += labels.size(0)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if batch_idx % 50 == 0:
            print('Epoch: [{}][{}/{}] '
                  'Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                  'lr:{:.3f} '
                  'Loss: {train_loss.val:.4f} ({train_loss.avg:.4f}) '
                  'iLoss: {id_loss.val:.4f} ({id_loss.avg:.4f}) '
                  'TLoss: {tri_loss.val:.4f} ({tri_loss.avg:.4f}) '
                  'Accu: {:.2f}'.format(
                epoch, batch_idx, len(trainloader), current_lr,
                100. * correct / total, batch_time=batch_time,
                train_loss=train_loss, id_loss=id_loss, tri_loss=tri_loss))

    writer.add_scalar('total_loss', train_loss.avg, epoch)
    writer.add_scalar('id_loss', id_loss.avg, epoch)
    writer.add_scalar('tri_loss', tri_loss.avg, epoch)
    writer.add_scalar('lr', current_lr, epoch)


def test(epoch):
    # switch to evaluation mode
    net.eval()
    print('Extracting Gallery Feature...')
    start = time.time()
    ptr = 0
    gall_feat = np.zeros((ngall, 2048*4))
    gall_feat_att = np.zeros((ngall, 2048*4))
    Xgall_feat = np.zeros((ngall, 2048*4))
    Xgall_feat_att = np.zeros((ngall, 2048*4))
    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(gall_loader):
            batch_num = input.size(0)
            input = Variable(input.cuda())
            feat, feat_att = net(input, input, test_mode[0])
            gall_feat[ptr:ptr + batch_num, :] = feat[:batch_num].detach().cpu().numpy()
            gall_feat_att[ptr:ptr + batch_num, :] = feat_att[:batch_num].detach().cpu().numpy()
            Xgall_feat[ptr:ptr + batch_num, :] = feat[batch_num:].detach().cpu().numpy()
            Xgall_feat_att[ptr:ptr + batch_num, :] = feat_att[batch_num:].detach().cpu().numpy()
            ptr = ptr + batch_num
    print('Extracting Time:\t {:.3f}'.format(time.time() - start))

    # switch to evaluation
    net.eval()
    print('Extracting Query Feature...')
    start = time.time()
    ptr = 0
    query_feat = np.zeros((nquery, 2048*4))
    query_feat_att = np.zeros((nquery, 2048*4))
    Xquery_feat = np.zeros((nquery, 2048*4))
    Xquery_feat_att = np.zeros((nquery, 2048*4))
    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(query_loader):
            batch_num = input.size(0)
            input = Variable(input.cuda())
            feat, feat_att = net(input, input, test_mode[1])
            query_feat[ptr:ptr + batch_num, :] = feat[:batch_num].detach().cpu().numpy()
            query_feat_att[ptr:ptr + batch_num, :] = feat_att[:batch_num].detach().cpu().numpy()
            Xquery_feat[ptr:ptr + batch_num, :] = feat[batch_num:].detach().cpu().numpy()
            Xquery_feat_att[ptr:ptr + batch_num, :] = feat_att[batch_num:].detach().cpu().numpy()
            ptr = ptr + batch_num
    print('Extracting Time:\t {:.3f}'.format(time.time() - start))

    start = time.time()
    # compute the similarity
    distmat = np.matmul(query_feat, np.transpose(gall_feat))
    distmat_att = np.matmul(query_feat_att, np.transpose(gall_feat_att))
    
    Xdistmat = np.matmul(query_feat, np.transpose(Xgall_feat))
    Xdistmat_att = np.matmul(query_feat_att, np.transpose(Xgall_feat_att))
    
    distmatX = np.matmul(Xquery_feat, np.transpose(gall_feat))
    distmat_attX = np.matmul(Xquery_feat_att, np.transpose(gall_feat_att))
    
    XXdistmat = np.matmul(Xquery_feat, np.transpose(Xgall_feat))
    XXdistmat_att = np.matmul(Xquery_feat_att, np.transpose(Xgall_feat_att))
    # evaluation

    cmc, mAP, mINP = eval_regdb(-distmat, query_label, gall_label)
    cmc_att, mAP_att, mINP_att = eval_regdb(-distmat_att, query_label, gall_label)

    Xcmc, XmAP, XmINP = eval_regdb(-Xdistmat, query_label, gall_label)
    Xcmc_att, XmAP_att, XmINP_att = eval_regdb(-Xdistmat_att, query_label, gall_label)

    cmcX, mAPX, mINPX = eval_regdb(-distmatX, query_label, gall_label)
    cmc_attX, mAP_attX, mINP_attX = eval_regdb(-distmat_attX, query_label, gall_label)

    XXcmc, XXmAP, XXmINP = eval_regdb(-XXdistmat, query_label, gall_label)
    XXcmc_att, XXmAP_att, XXmINP_att = eval_regdb(-XXdistmat_att, query_label, gall_label)
    print('Evaluation Time:\t {:.3f}'.format(time.time() - start))

    return cmc, mAP, mINP, cmc_att, mAP_att, mINP_att, \
    Xcmc, XmAP, XmINP, Xcmc_att, XmAP_att, XmINP_att, \
    cmcX, mAPX, mINPX, cmc_attX, mAP_attX, mINP_attX, \
    XXcmc, XXmAP, XXmINP, XXcmc_att, XXmAP_att, XXmINP_att

# training
print('==> Start Training...')
start_epoch = 0
for epoch in range(start_epoch, 81 - start_epoch):

    print('==> Preparing Data Loader...')
    # identity sampler
    sampler = IdentitySampler(trainset.train_color_label, \
                              trainset.train_thermal_label, color_pos, thermal_pos, args.num_pos, args.batch_size,
                              epoch)

    trainset.cIndex = sampler.index1  # color index
    trainset.tIndex = sampler.index2  # thermal index
    print(epoch)
    print(trainset.cIndex)
    print(trainset.tIndex)

    loader_batch = args.batch_size * args.num_pos

    trainloader = data.DataLoader(trainset, batch_size=loader_batch, \
                                  sampler=sampler, num_workers=args.workers, drop_last=True)

    # training
    train(epoch)

    if epoch > 0 and epoch % 2 == 0:
        print('Test Epoch: {}'.format(epoch))
    
        # testing
        cmc, mAP, mINP, cmc_att, mAP_att, mINP_att, \
        Xcmc, XmAP, XmINP, Xcmc_att, XmAP_att, XmINP_att, \
        cmcX, mAPX, mINPX, cmc_attX, mAP_attX, mINP_attX, \
        XXcmc, XXmAP, XXmINP, XXcmc_att, XXmAP_att, XXmINP_att = test(epoch)
        # save model
        if cmc_att[0] > best_acc:  # not the real best for sysu-mm01
            best_acc = cmc_att[0]
            best_epoch = epoch
            state = {
                'net': net.state_dict(),
                'cmc': cmc_att,
                'mAP': mAP_att,
                'mINP': mINP_att,
                'epoch': epoch,
            }
            torch.save(state, checkpoint_path + suffix + '_best.t')
    
        # save model
        if epoch > 10 and epoch % args.save_epoch == 0:
            state = {
                'net': net.state_dict(),
                'cmc': cmc,
                'mAP': mAP,
                'epoch': epoch,
            }
            torch.save(state, checkpoint_path + suffix + '_epoch_{}.t'.format(epoch))
    
        print('POOL:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
            cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))
        print('FC:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
            cmc_att[0], cmc_att[4], cmc_att[9], cmc_att[19], mAP_att, mINP_att))
        print('POOL:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
            Xcmc[0], Xcmc[4], Xcmc[9],Xcmc[19], XmAP, XmINP))
        print('FC:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
            Xcmc_att[0], Xcmc_att[4], Xcmc_att[9], Xcmc_att[19], XmAP_att, XmINP_att))
        print('POOL:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
            cmcX[0], cmcX[4], cmcX[9], cmcX[19], mAPX, mINPX))
        print('FC:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
            cmc_attX[0], cmc_attX[4], cmc_attX[9], cmc_attX[19], mAP_attX, mINP_attX))
        print('POOL:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
            XXcmc[0], XXcmc[4], XXcmc[9], XXcmc[19], XXmAP, XXmINP))
        print('FC:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
            XXcmc_att[0], XXcmc_att[4], XXcmc_att[9], XXcmc_att[19], XXmAP_att, XXmINP_att))
        print('Best Epoch [{}]'.format(best_epoch))