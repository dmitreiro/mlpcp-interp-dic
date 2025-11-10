# script ment to be run after abaqus cruciform-dic.py to perform DIC analysis using MatchID
# run on windows system with matchidstereo.exe in PATH
# run in the parent directory containing subdirectories with FEA results and Cruciform_FEDEF.mtind and Cruciform_DIC.m3inp files
import os
import shutil
import logging

current_dir = os.getcwd() # gets current directory

# ---- SETUP LOGGING ----
logfile = os.path.join(current_dir, "dic_log.txt")

logging.basicConfig(
    filename=logfile,
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
log = logging.getLogger()
log.info("===== DIC Automated Script Started =====")
log.info(f"Working directory: {current_dir}")

fedef_file = os.path.join(current_dir, "Cruciform_FEDEF.mtind")
dic_file = os.path.join(current_dir, "Cruciform_DIC.m3inp")
caldat = os.path.join(current_dir, "Calibration_data.caldat")
tiff_0 = os.path.join(current_dir, "Static_0000_0.tiff") # master camera
tiff_1 = os.path.join(current_dir, "Static_0000_1.tiff")

base_dir = os.path.join(current_dir, "_base")

entries = [
    d for d in os.listdir(current_dir)
    if os.path.isdir(os.path.join(current_dir, d)) and d != "_base"
]
total_samples = len(entries)

for index, name in enumerate(entries):
    path = os.path.join(current_dir, name)
    if path == base_dir:
        continue

    if os.path.isdir(path):
        log.info("Processing sample {} of {}".format(index + 1, total_samples))
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

        # run matchID
        os.chdir(base_dir)# change to working directory
        log.info("Running MatchID synthetic image creation...")
        os.system("matchidstereo.exe Cruciform_FEDEF.mtind")

        log.info("Running MatchID DIC analysis...")
        os.system("matchidstereo.exe Cruciform_DIC.m3inp")

        log.info("Copying results...")
        for f in os.listdir("."):
            if f.endswith(".csv") or f.endswith(".mesh"):
                try:
                    shutil.copy2(f, os.path.join(path, f))
                except Exception as e:
                    log.error(f"Error copying result file {f}: {e}")

        log.info("Cleaning up files...")
        for f in os.listdir("."):
            try:
                os.remove(f)
            except Exception as e:
                log.error(f"Error removing file {f}: {e}")

        log.info(f"Finished processing sample\n")        
        os.chdir(current_dir) # go back to parent directory

log.info("===== DIC Automated Script Finished =====")
