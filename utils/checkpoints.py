import pickle, imp, time
import gzip, logging, operator, os
import os.path as osp
import numpy as np
from path import Path
import coloredlogs


def convert2dict(params):
    names = [par.name for par in params]
    assert len(names) == len(set(names))

    param_dict = {par.name: par.get_value() for par in params}
    return param_dict


def save_weights(fname, params, history=None):
    param_dict = convert2dict(params)

    logging.info('saving {} parameters to {}'.format(len(params), fname))
    fname = Path(fname)

    filename, ext = osp.splitext(fname)
    history_file = osp.join(osp.dirname(fname), 'history.npy')
    np.save(history_file, history)
    logging.info("Save history to {}".format(history_file))
    if ext == '.npy':
        np.save(filename + '.npy', param_dict)
    else:
        f = gzip.open(fname, 'wb')
        pickle.dump(param_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.close()


def load_dict(fname):
    logging.info("Loading weights from {}".format(fname))
    filename, ext = os.path.splitext(fname)
    if ext == '.npy':
        params_load = np.load(fname).item()
    else:
        f = gzip.open(fname, 'r')
        params_load = pickle.load(f)
        f.close()
    if type(params_load) is dict:
        param_dict = params_load
    else:
        param_dict = convert2dict(params_load)
    return param_dict

def load_weights_trainable(fname, l_out):
    import lasagne
    params = lasagne.layers.get_all_params(l_out, trainable=True)
    names = [par.name for par in params]
    assert len(names) == len(set(names))

    if type(fname) is list:
        param_dict = {}
        for name in fname:
            t_load = load_dict(name)
            param_dict.update(t_load)
    else:
        param_dict = load_dict(fname)

    for param in params:
        if param.name in param_dict:
            stored_shape = np.asarray(param_dict[param.name].shape)
            param_shape = np.asarray(param.get_value().shape)
            if not np.all(stored_shape == param_shape):
                warn_msg = 'shape mismatch:'
                warn_msg += '{} stored:{} new:{}'.format(
                    param.name, stored_shape, param_shape)
                warn_msg += ', skipping'
                logging.warn(warn_msg)
            else:
                param.set_value(param_dict[param.name])
        else:
            logging.warn('unable to load parameter {} from {}: No such variable.'
                         .format(param.name, fname))



def load_weights(fname, l_out):
    import lasagne
    params = lasagne.layers.get_all_params(l_out)
    names = [par.name for par in params]
    assert len(names) == len(set(names))

    if type(fname) is list:
        param_dict = {}
        for name in fname:
            t_load = load_dict(name)
            param_dict.update(t_load)
    else:
        param_dict = load_dict(fname)
    assign_weights(params, param_dict)

def assign_weights(params, param_dict):
    for param in params:
        if param.name in param_dict:
            stored_shape = np.asarray(param_dict[param.name].shape)
            param_shape = np.asarray(param.get_value().shape)
            if not np.all(stored_shape == param_shape):
                warn_msg = 'shape mismatch:'
                warn_msg += '{} stored:{} new:{}'.format(
                    param.name, stored_shape, param_shape)
                warn_msg += ', skipping'
                logging.warn(warn_msg)
            else:
                param.set_value(param_dict[param.name])
        else:
            logging.warn('Unable to load parameter {}: No such variable.'
                         .format(param.name))


def get_list_name(obj):
    if type(obj) is list:
        for i in range(len(obj)):
            if callable(obj[i]):
                obj[i] = obj[i].__name__
    elif callable(obj):
        obj = obj.__name__
    return obj


# write commandline parameters to header of logfile
def build_log_file(cfg):
    FORMAT="%(asctime)s;%(levelname)s|%(message)s"
    DATEF="%H-%M-%S"
    logging.basicConfig(formatter=FORMAT, level=logging.DEBUG)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler(filename=os.path.join(cfg['outfolder'], 'logfile'+time.strftime("%m-%d")+'.log'))
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s;%(levelname)s|%(message)s", "%H:%M:%S")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    LEVEL_STYLES = dict(
        debug=dict(color='magenta'),
        info=dict(color='green'),
        verbose=dict(),
        warning=dict(color='blue'),
        error=dict(color='yellow'),
        critical=dict(color='red',bold=True))
    coloredlogs.install(level=logging.DEBUG, fmt=FORMAT, datefmt=DATEF, level_styles=LEVEL_STYLES)


    args_dict = cfg
    sorted_args = sorted(args_dict.items(), key=operator.itemgetter(0))
    logging.info('######################################################')
    logging.info('# --Configurable Parameters In this Model--')
    for name, val in sorted_args:
        logging.info("# " + name + ":\t" + str(get_list_name(val)))
    logging.info('######################################################')


def get_cfg(args):
    if args.cfg is not None:
        cfg = imp.load_source('config', args.cfg)
    else:
        raise Exception("The file path of config_file cannot be ignored")

    getmodel = cfg.get_model
    cfg = cfg.cfg
    args = vars(args).items()
    for name, val in args:
        cfg[name] = val

    cfg['outfolder'] = os.path.join(cfg['outfolder'], cfg['name'])
    res_out = cfg['outfolder']
    if 'key_point' in cfg:
        res_out += '.'+ cfg['key_point']
        if cfg['key_point'] in cfg:
            res_out += '-' + str(cfg[cfg['key_point']])
    if 'notime' not in cfg or cfg['notime'] in [False, 'False', 'false', None, 'none', 'None']:
        res_out += '.' + time.strftime("%b-%d--%H-%M")

    res_out = os.path.realpath(res_out)
    if os.path.exists(res_out):
        tcount = 1
        while os.path.exists(res_out+'+'+str(tcount)):
            tcount += 1
        res_out += '+' + str(tcount)

    # print res_out
    os.makedirs(res_out)
    cfg['outfolder'] = res_out
    return cfg, getmodel
