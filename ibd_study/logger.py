import datetime
import os
import logging

log_file_name = "model_run.log"

# Set up the logger's basic configuration.
logging.basicConfig(
    filename=log_file_name,  # The file to write the log to.
    level=logging.INFO,      # The minimum level of messages to log.
    format='%(asctime)s - %(levelname)s - %(message)s' # The format of each log message.
)
logging.getLogger('hyperopt').propagate = False