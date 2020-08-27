import time
import logging


def timeit(method):

    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()

        logging.info('%s %2.2f sec' %
                     (method.__name__.upper(), te-ts))
        return result

    return timed
