# script ment to be run after abaqus cruciform-dic.py to perform DIC analysis using MatchID
# run on windows system with matchidstereo.exe in PATH
# run in the parent directory containing subdirectories with FEA results and Cruciform_FEDEF.mtind and Cruciform_DIC.m3inp files
import os
import shutil

current_dir = os.getcwd() # gets current directory

fedef_file = os.path.join(current_dir, "Cruciform_FEDEF.mtind")
dic_file = os.path.join(current_dir, "Cruciform_DIC.m3inp")
caldat = os.path.join(current_dir, "Calibration_data.caldat")
tiff_0 = os.path.join(current_dir, "Static_0000_0.tiff") # master camera
tiff_1 = os.path.join(current_dir, "Static_0000_1.tiff")

base_dir = os.path.join(current_dir, "_base")

for name in os.listdir(current_dir):
    path = os.path.join(current_dir, name)
    if path == base_dir:
        continue
    elif os.path.isdir(path):
        # copy files to directory
        shutil.copy(fedef_file, base_dir)
        shutil.copy(dic_file, base_dir)
        shutil.copy(caldat, base_dir)
        shutil.copy(tiff_0, base_dir)
        shutil.copy(tiff_1, base_dir)

        # Loop through all files inside this folder
        for item in os.listdir(path):
            src_path = os.path.join(path, item)
            dst_path = os.path.join(base_dir, item)

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

        os.chdir(base_dir) # change to working directory

        os.system("matchidstereo.exe Cruciform_FEDEF.mtind") # create synthetic images
        os.system("matchidstereo.exe Cruciform_DIC.m3inp") # executes DIC analysis

        # move results to original directory
        for f in os.listdir("."):
            if f.endswith(".csv") \
            or f.endswith(".mesh"):
                shutil.copy(f, os.path.join(path, f))

        # clean up files
        for f in os.listdir("."):
            os.remove(f)
        
        os.chdir(current_dir) # go back to parent directory