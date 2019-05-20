import os
import pickle
from config import config as cfg
from tensorpack import logger

# def load_class_names(name, path=''):
#     '''
#     '''
#     if not path:
#         path = os.path.dirname(__file__)
#     fn_names = os.path.join(path, 'names', name + '.names')
#     cat_names = []
#     with open(fn_names, 'r') as fh:
#         cat_names = [l.strip() for l in fh.readlines()]
#     cat_ids = list(range(len(cat_names)))
#     return cat_names, cat_ids
#
#
# def cname2cid(cname, class_names):
#     '''
#     convert a class name to class id, handling ignore classes
#     Arg:
#         cname: class name to convert
#         class_names: all the class names in a list
#
#     Return:
#         * class_names.index(cname), if cname in class_names.
#         * -1, if cname is in ignore class list.
#     '''
#     if cname.lower() in cfg.DATA.IGNORE_NAMES:
#         return -1
#     return class_names.index(cname.lower())


# def recursive_parse_xml_to_dict(xml):
#   """Recursively parses XML contents to python dict.
#
#   We assume that `object` tags are the only ones that can appear
#   multiple times at the same level of a tree.
#
#   Args:
#     xml: xml tree obtained by parsing XML file contents using lxml.etree
#
#   Returns:
#     Python dictionary holding XML contents.
#   """
#   if not len(xml):
#     return {xml.tag: xml.text}
#   result = {}
#   for child in xml:
#     child_result = recursive_parse_xml_to_dict(child)
#     if child.tag not in ('object', 'part'):
#       result[child.tag] = child_result[child.tag]
#     else:
#       if child.tag not in result:
#         result[child.tag] = []
#       result[child.tag].append(child_result[child.tag])
#   return {xml.tag: result}


def save_to_cache(dataset, name):
    fn_cache = os.path.join(cfg.DATA.CACHEDIR, name + '.pkl')
    with open(fn_cache, 'wb') as fh:
        pickle.dump({'dataset': dataset}, fh)


def load_from_cache(name, ctime=0):
    fn_cache = os.path.join(cfg.DATA.CACHEDIR, name + '.pkl')
    if ctime > os.path.getmtime(fn_cache):
        logger.warn('cache file is older than the dataset file')
        # raise IOError('cache file is older than the dataset file')
    try:
        with open(fn_cache, 'rb') as fh:
            dataset = pickle.load(fh)['dataset']
    except:
        raise IOError
    return dataset


def load_many_from_db(dataset, add_gt, is_train):
    '''
    '''
    klass_name = '{}Segmentation'.format(dataset.upper())
    module = __import__('dataset.{}'.format(dataset), fromlist=[klass_name])
    klass = getattr(module, klass_name)

    db_cfg = getattr(cfg.DATA, dataset.upper())
    db_type = 'TRAIN' if is_train else 'TEST'
    return klass.load_many(db_cfg.BASEDIR, getattr(db_cfg, db_type), add_gt)


def clip_bbox(bb, frame):
    '''
    clip bb into frame
    bb and frame: [x0, y0, x1, y1]
    '''
    x0 = max(frame[0], min(frame[2], bb[0]))
    y0 = max(frame[1], min(frame[3], bb[1]))
    x1 = max(frame[0], min(frame[2], bb[2]))
    y1 = max(frame[1], min(frame[3], bb[3]))
    return [x0, y0, x1, y1]
