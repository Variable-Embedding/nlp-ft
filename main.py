"""Creating and executing a pipeline.
"""
from src.util.configuration import run_configuration
from src.stages.stage_factory import create_pipeline_from_config

import argparse
import logging

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Running a pipeline.')
    parser.add_argument('--config-file', action='append', nargs='+')
    parser.add_argument('--topic', nargs='?', default="countries")
    args = parser.parse_args()

    pywiki_logger = logging.getLogger("pywiki")

    run_configuration()
    for config_file in args.config_file:
        pipeline = create_pipeline_from_config(config_file[0], topic=args.topic)
        pipeline.execute()
