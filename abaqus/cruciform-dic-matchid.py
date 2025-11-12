# script ment to be run after abaqus cruciform-dic.py to perform DIC analysis using MatchID
# run on windows system with matchidstereo.exe in PATH
# run in the parent directory containing subdirectories with FEA results and Cruciform_FEDEF.mtind and Cruciform_DIC.m3inp files
import os
import shutil
import logging
import telebot
import configparser

current_dir = os.getcwd() # gets current directory
base_dir = os.path.join(current_dir, "_base")

fedef_file = os.path.join(current_dir, "Cruciform_FEDEF.mtind")
dic_file = os.path.join(current_dir, "Cruciform_DIC.m3inp")
caldat = os.path.join(current_dir, "Calibration_data.caldat")
tiff_0 = os.path.join(current_dir, "Static_0000_0.tiff") # master camera
tiff_1 = os.path.join(current_dir, "Static_0000_1.tiff")

# logfile = os.path.join(current_dir, "dic_log.txt")
logfile = r"Z:\dic_log.txt"  # hardcoded path

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
    config.read(os.path.join(current_dir, "config.ini"))
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

# ---- MAIN SCRIPT ----
def main():

    log.info("===== DIC Automated Script Started =====")
    log.info(f"Working directory: {current_dir}")
    ntfy("DIC Automated Script Started")

    entries = [
        d for d in os.listdir(current_dir)
        if os.path.isdir(os.path.join(current_dir, d)) and d != "_base"
    ]
    total_samples = len(entries)

    for index, name in enumerate(entries, start=1):
        path = os.path.join(current_dir, name)
        if path == base_dir:
            continue

        if os.path.isdir(path):
            log.info("Processing sample {} of {}".format(index, total_samples))
            log.info("Directory: {}".format(name))

            # copy base files
            try:
                shutil.copy(fedef_file, base_dir)
                shutil.copy(dic_file, base_dir)
                shutil.copy(caldat, base_dir)
                shutil.copy(tiff_0, base_dir)
                shutil.copy(tiff_1, base_dir)
                log.info("Copied MatchID files to _base")
            except Exception as e:
                log.error(f"Error copying base files: {e}")
                ntfy(f"Error copying base files for sample {name}: {e}")
                exit(1)

            log.info("Copying sample files to _base...")
            # Loop through all files inside this folder
            for item in os.listdir(path):
                src_path = os.path.join(path, item)
                dst_path = os.path.join(base_dir, item)

                try:
                    # Copy file or folder
                    if os.path.isfile(src_path):
                        shutil.copy2(src_path, dst_path)
                    elif os.path.isdir(src_path):
                        # Copy folder without overwriting if it already exists
                        if not os.path.exists(dst_path):
                            shutil.copytree(src_path, dst_path)
                        else:
                            # Folder exists → copy contents inside it
                            for subitem in os.listdir(src_path):
                                shutil.copy2(
                                    os.path.join(src_path, subitem),
                                    os.path.join(dst_path, subitem)
                                )
                except Exception as e:
                    log.error(f"Error copying {src_path}: {e}")
                    ntfy(f"Error copying src_files for sample {name}: {e}")
                    exit(1)

            # run matchID
            os.chdir(base_dir)# change to working directory
            log.info("Running MatchID synthetic image creation...")
            fdef_stat = os.system("matchidstereo.exe Cruciform_FEDEF.mtind")
            if fdef_stat != 0:
                log.error(f"matchidstereo.exe failed with exit code {fdef_stat}")
                ntfy(f"matchidstereo.exe failed with exit code {fdef_stat} for sample {name}")
                exit(1)

            log.info("Running MatchID DIC analysis...")
            dic_stat = os.system("matchidstereo.exe Cruciform_DIC.m3inp")
            if dic_stat != 0:
                log.error(f"matchidstereo.exe failed with exit code {dic_stat}")
                ntfy(f"matchidstereo.exe failed with exit code {dic_stat} for sample {name}")
                exit(1)
            
            log.info("Copying results...")
            for f in os.listdir("."):
                if f.endswith(".csv") or f.endswith(".mesh"):
                    try:
                        shutil.copy2(f, os.path.join(path, f))
                    except Exception as e:
                        log.error(f"Error copying result file {f}: {e}")
                        ntfy(f"Error copying result file {f} for sample {name}: {e}")
                        exit(1)

            log.info("Cleaning up files...")
            for f in os.listdir("."):
                try:
                    os.remove(f)
                except Exception as e:
                    log.error(f"Error removing file {f}: {e}")
                    ntfy(f"Error removing file {f} for sample {name}: {e}")
                    exit(1)

            log.info(f"Finished processing sample\n")
            os.chdir(current_dir) # go back to parent directory
        
        # telegram notification every 100 samples
        if index % 100 == 0:
            ntfy("Processed {} of {}".format(index, total_samples))

    log.info("===== DIC Automated Script Finished =====")
    ntfy("DIC Automated Script Finished")
    exit(0)

if __name__ == "__main__":
    main()