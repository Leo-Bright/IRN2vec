import json
import models
import logger
import inspect
import datetime
import argparse

parser = argparse.ArgumentParser()
# basic args
parser.add_argument('--task', type = str, default='train')
parser.add_argument('--batch_size', type = int, default = 128)
parser.add_argument('--epochs', type = int, default = 1)

# train args
parser.add_argument('--train', type = str)
parser.add_argument('--type_file', type = str)
parser.add_argument('--alpha', type = float, default=0.025)
parser.add_argument('--beta', type = float, default=0.8)
parser.add_argument('--output', type = str)
parser.add_argument('--negative', type = int)
parser.add_argument('--threads', type = int, default=1)

# log file name
parser.add_argument('--log_file', type = str, default='./train_log.log')

args = parser.parse_args()


def train(model, elogger, train_set, eval_set):
    pass
    # record the experiment setting
    # elogger.log(str(model))
    # elogger.log(str(args._get_kwargs()))
    #
    # model.train()
    #
    # if torch.cuda.is_available():
    #     model.cuda()
    #
    # optimizer = optim.Adam(model.parameters(), lr = 1e-3)
    #
    # for epoch in xrange(args.epochs):
    #     print('Training on epoch {}'.format(epoch))
    #     for input_file in train_set:
    #         print('Train on file {}'.format(input_file))
    #
    #         # data loader, return two dictionaries, attr and traj
    #         data_iter = data_loader.get_loader(input_file, args.batch_size)
    #
    #         running_loss = 0.0
    #
    #         for idx, (attr, traj) in enumerate(data_iter):
    #             # transform the input to pytorch variable
    #             attr, traj = utils.to_var(attr), utils.to_var(traj)
    #
    #             _, loss = model.eval_on_batch(attr, traj, config)
    #
    #             # update the model
    #             optimizer.zero_grad()
    #             loss.backward()
    #             optimizer.step()
    #
    #             # running_loss += loss.data[0]
    #             running_loss += loss.item()
    #             print '\r Progress {:.2f}%, average loss {}'.format((idx + 1) * 100.0 / len(data_iter), running_loss / (idx + 1.0)),
    #         print
    #         elogger.log('Training Epoch {}, File {}, Loss {}'.format(epoch, input_file, running_loss / (idx + 1.0)))
    #
    #     # evaluate the model after each epoch
    #     evaluate(model, elogger, eval_set, save_result = False)
    #
    #     # save the weight file after each epoch
    #     weight_name = '{}_{}'.format(args.log_file, str(datetime.datetime.now()))
    #     elogger.log('Save weight file {}'.format(weight_name))
    #     torch.save(model.state_dict(), './saved_weights/' + weight_name)


def write_result(fs, pred_dict, attr):
    pred = pred_dict['pred'].data.cpu().numpy()
    label = pred_dict['label'].data.cpu().numpy()

    for i in range(pred_dict['pred'].size()[0]):
        fs.write('%.6f %.6f\n' % (label[i][0], pred[i][0]))

        dateID = attr['dateID'].data[i]
        timeID = attr['timeID'].data[i]
        driverID = attr['driverID'].data[i]


def evaluate(model, elogger, files, save_result = False):
    pass
    # model.eval()
    # if save_result:
    #     fs = open('%s' % args.result_file, 'w')
    #
    # for input_file in files:
    #     running_loss = 0.0
    #     data_iter = data_loader.get_loader(input_file, args.batch_size)
    #
    #     # attr,traj in here are a batch size sample's
    #     for idx, (attr, traj) in enumerate(data_iter):
    #         attr, traj = utils.to_var(attr), utils.to_var(traj)
    #
    #         pred_dict, loss = model.eval_on_batch(attr, traj, config)
    #
    #         if save_result: write_result(fs, pred_dict, attr)
    #
    #         # running_loss += loss.data[0]
    #         running_loss += loss.item()
    #
    #     print 'Evaluate on file {}, loss {}'.format(input_file, running_loss / (idx + 1.0))
    #     elogger.log('Evaluate File {}, Loss {}'.format(input_file, running_loss / (idx + 1.0)))

    # if save_result: fs.close()


def get_kwargs(model_class):
    model_args = inspect.getargspec(model_class.__init__).args
    shell_args = args._get_kwargs()

    kwargs = dict(shell_args)

    for arg, val in shell_args:
        if not arg in model_args:
            kwargs.pop(arg)

    return kwargs


def run():
    # get the model arguments
    # arguments value from command line input
    kwargs = get_kwargs(models.IRN2vec.IRN2vec)

    # model instance
    model = models.IRN2vec.IRN2vec(**kwargs)

    # experiment logger
    elogger = logger.Logger(args.log_file)

    if args.task == 'train':
        train(model, elogger)

    elif args.task == 'test':
        # load the saved weight file
        model.load_state_dict(torch.load(args.weight_file))
        if torch.cuda.is_available():
            model.cuda()
        evaluate(model, elogger, config['test_set'], save_result = True)


if __name__ == '__main__':
    run()
