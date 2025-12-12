"""
Tool to compile csv files.
"""

import glob
import os
import csv
import pandas as pd
import re
# import time
import logging
import telebot
import configparser

# Accessing variables
DIR = r"data/raw"
DATA_PROC = r"data/processed"
TRAIN = os.path.join(DIR, "compilation", "train")
TEST = os.path.join(DIR, "compilation", "test")

# ensure output directory exists
os.makedirs(DATA_PROC, exist_ok=True)

# buffer threshold for dumping intermediate files
X_BUFF_TSHOLD = 100

# logfile = os.path.join(current_dir, "dic_log.txt")
logfile = r"Z:\dic_2nd_compilation_log.txt"  # hardcoded path

# ---- SETUP LOGGING ----
logging.basicConfig(
    filename=logfile,
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
log = logging.getLogger()

# ---- SETUP TELEGRAM CONFIG ----
config = configparser.ConfigParser()
try:
    config.read(r"config/config.ini")
    TOKEN = config.get("Telegram", "token")
    CHAT_ID = config.get("Telegram", "chat_id")
except Exception as e:
    log.error(f"Error reading configuration file: {e}")

def ntfy(msg: str) -> None:
    """
    Sends `msg` string to Telegram bot defined with `TOKEN` and `CHAT_ID` global variables.
    """
    # starting bot
    bot = telebot.TeleBot(TOKEN)

    try:
        bot.send_message(CHAT_ID, msg)
    except Exception as e:
        log.error(f"Failed to send Telegram notification: {e}")

def main():
    log.info("===== DIC 2nd compilation script started =====")
    ntfy("DIC 2nd compilation script started")

    for dataset_type in [TRAIN, TEST]:
        base_dir = dataset_type
        if dataset_type == TRAIN:
            X = os.path.join(DATA_PROC, "x_train_dic_compiled.csv")
            Y = os.path.join(DATA_PROC, "y_train_dic_compiled.csv")
        else:
            X = os.path.join(DATA_PROC, "x_test_dic_compiled.csv")
            Y = os.path.join(DATA_PROC, "y_test_dic_compiled.csv")
        
        log.info(f"Processing dataset: {os.path.basename(dataset_type)}")
        ntfy(f"Processing dataset: {os.path.basename(dataset_type)}")

        # Get all the csv files in that directory (assuming they have the extension .csv)
        csvfiles = glob.glob(os.path.join(base_dir, "*.csv"))
        total_files = len(csvfiles)

        # Checking for previous data files
        if os.path.isfile(X):
            os.remove(X)
        if os.path.isfile(Y):
            os.remove(Y)

        final_rows = []
        final_y = []

        for index, cs in enumerate(csvfiles, start=1):
            log.info(f"Reading file {cs}")
            rows = []
            try:
                with open(cs, "r") as file:
                    csvreader = csv.reader(file)
                    for row in csvreader:
                        rows.append(row)
                rows = [x for f in rows for x in f]
                final_rows.append(rows)
                final_y.append(re.findall(r"\d+(?:\.\d+)?", cs))
            except Exception as e:
                log.error(f"Error reading file {cs}: {e}")
                ntfy(f"Error reading file {cs}: {e}")
                continue  # skip to next file

            # progress
            log.info(f"Processed {index}/{total_files} files")

            # If buffer x file has more than X_BUFF_TSHOLD lines, it is dumped
            if len(final_rows) == X_BUFF_TSHOLD:
                # progress
                log.info("Dumping x buffer file")
                p = pd.DataFrame(final_rows)
                p.to_csv(X, mode="a", header=False, index=False)
                final_rows = []

        log.info("Dataframe x and y data")
        p = pd.DataFrame(final_rows)
        pf = pd.DataFrame(final_y, columns=["F", "G", "H", "L", "M", "N", "sigma0", "k", "n"])

        log.info(f"Writting final x and y data for {os.path.basename(dataset_type)}")
        ntfy(f"Writting final x and y data for {os.path.basename(dataset_type)}")
        p.to_csv(X, mode="a", header=False, index=False)
        pf.to_csv(Y)

        log.info(f"Dataset {os.path.basename(dataset_type)} compilation completed.")
        ntfy(f"Dataset {os.path.basename(dataset_type)} compilation completed.")

    log.info("===== DIC 2nd compilation script finished =====")
    ntfy("DIC 2nd compilation script finished")

if __name__ == "__main__":
    main()