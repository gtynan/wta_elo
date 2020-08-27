import sys
import getopt
import logging
from datetime import datetime

from src.pipeline import run

# log to file
logging.basicConfig(level=logging.DEBUG,
                    filename=f'logs/PIPELINE_RUN: {datetime.now()}.log',
                    format=' %(asctime)s - %(levelname)s - %(message)s',)


if __name__ == "__main__":
    # run()
    args = {arg: val for (arg, val) in getopt.getopt(sys.argv[1:], '', ['yf=', 'yt=', 'ts='])[0]}

    logging.debug(f'ARGS: {args}')

    run(year_from=int(args['--yf']),
        year_to=int(args['--yt']),
        test_size=int(args['--ts']))
