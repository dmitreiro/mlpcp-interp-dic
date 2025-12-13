import os
import csv
import logging
import telebot
import configparser

# Accessing variables
# base_dir = os.getcwd()
DIR = r"data/raw"
TRAIN = os.path.join(DIR, "train")
TEST = os.path.join(DIR, "test")
OUTPUT = os.path.join(DIR, "compilation")
ORIGINAL = os.path.join(DIR, "original_samples")
REF_FILE = os.path.join(DIR, "Static_0000_0_Numerical_0_0.synthetic.tif.csv")

# ensure output directory exists
os.makedirs(OUTPUT, exist_ok=True)

# logfile = os.path.join(current_dir, "dic_log.txt")
logfile = r"Z:\dic_1st_compilation_log.txt"  # hardcoded path

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
    log.info("===== DIC 1st compilation script started =====")
    ntfy("DIC 1st compilation script started")

    log.info("Reading reference file for row order...")
    # === read reference file (file 0) to define row order ===
    if not os.path.exists(REF_FILE):
        log.error(f"Missing reference file: {REF_FILE}")
        ntfy(f"Missing reference file: {REF_FILE}")
        exit(1)

    try:
        ref_rows = []
        with open(REF_FILE, newline='', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=';')
            next(reader, None)  # skip header

            for row in reader:
                # use (col0, col1) as sorting key
                key = (
                    float(row[0].replace(",", ".")),
                    float(row[1].replace(",", "."))
                )
                ref_rows.append(key)

        # build a lookup: coordinate -> index
        ref_index = {key: idx for idx, key in enumerate(ref_rows)}

    except Exception as e:
        log.error(f"Error reading reference file {REF_FILE}: {e}")
        ntfy(f"Error reading reference file {REF_FILE}: {e}")
        exit(1)

    skip_folder = False
    # run through both train and test directories
    for dataset_type in [TRAIN, TEST]:
        base_dir = dataset_type
        output_dir = os.path.join(OUTPUT, os.path.basename(dataset_type))

        # ensure output sub-directory exists
        os.makedirs(output_dir, exist_ok=True)

        log.info(f"Processing dataset: {os.path.basename(dataset_type)}")
        ntfy(f"Processing dataset: {os.path.basename(dataset_type)}")

        entries = [
            d for d in os.listdir(base_dir)
            if os.path.isdir(os.path.join(base_dir, d)) and d != "_base"
        ]
        total_samples = len(entries)

        for index, folder in enumerate(entries, start=1):
            folder_path = os.path.join(base_dir, folder)
            if not os.path.isdir(folder_path):
                continue
            if folder in ["_base"]:
                continue  # do not process special folders

            log.info("Processing sample {} of {}".format(index, total_samples))
            log.info("Directory: {}".format(folder))

            data = []

            # === read 20 synthetic files ===
            for i in range(1, 21):
                filename = f"Static_0000_0_Numerical_{i}_0.synthetic.tif.csv"
                file_path = os.path.join(folder_path, filename)

                if not os.path.exists(file_path):
                    log.error(f"Missing file: {filename} in folder {folder}")
                    ntfy(f"Missing file: {filename} in folder {folder}")
                    skip_folder = True
                    break

                # strains = []
                rows = []

                try:
                    with open(file_path, newline='', encoding='utf-8') as f:
                        reader = csv.reader(f, delimiter=';')
                        next(reader, None)  # skip header

                        for row in reader:
                            # strains.extend([row[16], row[17], row[18]])
                            key = (
                                float(row[0].replace(",", ".")),
                                float(row[1].replace(",", "."))
                            )
                            # convert decimal comma to float:
                            v1 = float(row[16].replace(",", "."))
                            v2 = float(row[17].replace(",", "."))
                            v3 = float(row[18].replace(",", "."))
                            # strains.extend([v1, v2, v3])
                            rows.append((key, [v1, v2, v3]))

                    # sort rows according to reference file order
                    rows.sort(key=lambda x: ref_index[x[0]])

                    # flatten strains in sorted order
                    strains = []
                    for _, vals in rows:
                        strains.extend(vals)

                    data.append(strains)

                except Exception as e:
                    log.error(f"Error reading file {filename} in folder {folder}: {e}")
                    ntfy(f"Error reading file {filename} in folder {folder}: {e}")
                    skip_folder = True
                    break

            if skip_folder:
                skip_folder = False
                continue  # skip to next folder

            # === read original samples ===
            original_file = os.path.join(ORIGINAL, f"{folder}.csv")

            if not os.path.exists(original_file):
                log.error(f"Missing original sample file: {folder}.csv")
                ntfy(f"Missing original sample file: {folder}.csv")
                continue

            try:
                with open(original_file, newline='', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    for i, row in enumerate(reader):
                        # prepend first 2 values
                        data[i] = [row[0], row[1]] + data[i]
            except Exception as e:
                log.error(f"Error reading original sample file {folder}.csv: {e}")
                ntfy(f"Error reading original sample file {folder}.csv: {e}")
                continue

            # === save final compiled CSV ===
            output_file = os.path.join(output_dir, f"{folder}.csv")
            try:
                with open(output_file, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerows(data)
            except Exception as e:
                log.error(f"Error writing compiled file for {folder}: {e}")
                ntfy(f"Error writing compiled file for {folder}: {e}")
                continue

        log.info(f"Dataset {os.path.basename(dataset_type)} compilation completed.")
        ntfy(f"Dataset {os.path.basename(dataset_type)} compilation completed.")

    log.info("===== DIC 1st compilation script finished =====")
    ntfy("DIC 1st compilation script finished")
    exit(0)

if __name__ == "__main__":
    main()