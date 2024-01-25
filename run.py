import argparse
import os
import numpy as np
import random
import pandas as pd

import torch
import torch.nn as nn
import yaml

from utils.log_helper import *
from utils.data_helper import *
from utils.eval_helper import evaluate
from utils.train_helper import *
from models.model import *
from loss.loss import *

def run(args, config_dict):
    """Main 

    Args:
        args (_type_): _description_
        config_dict (_type_): _description_
    """

    # Create Log file
    key = config_dict['loop']
    idx = '-' + str(config_dict[key])
    log_save_id = create_log_id(config_dict['log_name'] + '-' + config_dict['note'] + '-' +idx, config_dict['log_dir'])
    logging_config(folder = config_dict['log_dir'], name='{}'.format(log_save_id), no_console=False)
    logging.info(args)
    logging_dict(config_dict)


    # Create Saved model
    checkpoint = create_save_checkpoint(log_save_id, config_dict['save_dir'])

    # Initialize Random Seed
    def set_seed(seed: int = 42) -> None:
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        # When running on the CuDNN backend, two further options must be set
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Set a fixed value for the hash seed
        os.environ["PYTHONHASHSEED"] = str(seed)
        print(f"Random seed set as {seed}")
        
    set_seed(config_dict['seed'])  
    g = torch.Generator()
    g.manual_seed(0)
    
    
    
    # read data 
    data_reader = Data_Reader(config_dict)
    
    dl = eval(config_dict['dataloader'])
    train_dataloader = dl(data_reader, config_dict, g)

    device = torch.device("cuda:%d"%(args.gpu) if torch.cuda.is_available() else "cpu")
    config_dict['device'] = device

    fn = eval(config_dict['loss'])

    loss_fn = fn(config_dict) 

    ml = eval(config_dict['model'])
    
    if config_dict['model'] in {'LightGCN', 'LightGCN_Debiased'}:
        model = ml(data_reader.num_user, 
                   data_reader.num_item, 
                   config_dict,
                   data_reader.getSparseGraph(device)
                  ).to(device)
    elif config_dict['model'] in {'MF', 'MF_Debiased'}:
         model = ml(data_reader.num_user, 
                   data_reader.num_item, 
                   config_dict).to(device)

    
    optimizer = torch.optim.Adam(model.parameters(), lr = config_dict['learning_rate'])

    # initialize some monitors and flags
    best_monitor = 0
    best_epoch = 0

    model.eval()
    logging.info("Evaluating at initialization")
    res = evaluate(config_dict, model, data_reader, config_dict['metrics'], mode = 'test', normalize = config_dict['cosine_flag_test'])
    
    stopping_steps = 0
    # train model and eval
    for epoch in range(config_dict['epochs']):

        model.train()
        logging.info("Training..")
        losses = []
        for idx, batch_data in enumerate(tqdm(train_dataloader)):
            
            
            user = batch_data['user'].long().to(device)
            pos_item = batch_data['pos_item'].long().to(device)
            neg_item = batch_data['neg_item'].long().to(device)

            model.zero_grad()
            # Debiased loss
            if config_dict['dataloader'] == 'get_Decoupling_CCL_trainloader_Debiased':
                extra_item = batch_data['extra_item'].long().to(device)
                tu = batch_data['tu'].float().to(device)
                
                pos_scores,  neg_scores, extra_scores, reg = model.compute(user, pos_item, neg_item, extra_item)           
                loss = loss_fn(pos_scores, neg_scores, extra_scores, tu)
            # Non-debiased loss
            else:
                pos_scores,  neg_scores, reg = model.compute(user, pos_item, neg_item)
                loss = loss_fn(pos_scores, neg_scores)
            
            loss += (config_dict['regularizer'] / 2) * reg            
       
            loss.backward()

            optimizer.step()
            losses.append(loss.item())


        model.eval()
        logging.info("Evaluating..")
        res = evaluate(config_dict, model, data_reader, config_dict['metrics'], mode = 'valid', normalize = config_dict['cosine_flag_test'])
        logging.info("Avg batch loss = {}".format(np.mean(losses)))
        
        
        
        # EARLY STOP, SVAE MODEL, LOAD MODEL
        monitor_value = res[config_dict['monitor']]
        if monitor_value > best_monitor:
            if config_dict['reset_lr']:
                if stopping_steps > 0:
                    reset_learning_rate(config_dict, optimizer)
                    logging.info("Reset learning rate to {:.6f}".format(config_dict['learning_rate']))
            stopping_steps = 0 
            best_monitor = monitor_value
            best_metric = res
            
            best_epoch = epoch
            torch.save(model.state_dict(), checkpoint)
            
        else:
            stopping_steps += 1
            logging.info("Monitor({}) STOP: {:.6f} !".format(config_dict['monitor'], monitor_value))

            # reduce learning rate and load best model
            current_lr = reduce_learning_rate(optimizer, factor = config_dict['lr_reduce_factor'])
            logging.info("Reduce learning rate on plateau: {:.6f}".format(current_lr))
                
            model.load_state_dict(torch.load(checkpoint, map_location=device))
            logging.info("Load best model from: {}".format(checkpoint))
        logging.info(f"best {config_dict['monitor']}= {best_monitor} at epoch {best_epoch}")

        if stopping_steps  >= config_dict['patience']:
            logging.info("Early stopping at epoch={:g}".format(epoch))
            break
        else:
            logging.info("************ Epoch={} end ************".format(epoch))
        
    logging.info('--------Finish Training--------')      
    logging.info(f"best_epoch is {best_epoch}")
    logging.info(best_metric)
    
    logging.info('--------Start Final Test--------')      
    model.eval()
    logging.info("Testing..")
    res = evaluate(config_dict, model, data_reader, config_dict['metrics'], mode = 'test', normalize = config_dict['cosine_flag_test'])
    logging.info("Final Test Finished. All Done!!!")

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--note', type=str, default='')
    parser.add_argument('--config_path', type=str, default='./')
    parser.add_argument('--gpu', type = int, default = '0')
    parser.add_argument('--key', type = float, default = 1)
    


    args = parser.parse_args()
    
    with open(args.config_path, 'r') as stream:
        config_dict = yaml.safe_load(stream)

    config_dict.update(vars(args)) 

    config_dict['seed'] = int(config_dict['seed'])

    run(args, config_dict)


