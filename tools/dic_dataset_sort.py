import csv
import os

# Accessing variables
DATA_PROC = r"data/processed"
DATA_CLEAN = r"data/cleaned"
Y_TRAIN_OLD = os.path.join(DATA_CLEAN, "y_train.csv")
Y_TRAIN = os.path.join(DATA_PROC, "y_train_dic_compiled.csv")
X_TRAIN = os.path.join(DATA_PROC, "x_train_dic_compiled.csv")
NEW_Y_TRAIN = os.path.join(DATA_CLEAN, "y_train_dic.csv")
NEW_X_TRAIN = os.path.join(DATA_CLEAN, "x_train_dic.csv")

Y_TEST_OLD = os.path.join(DATA_CLEAN, "y_test.csv")
Y_TEST = os.path.join(DATA_PROC, "y_test_dic_compiled.csv")
X_TEST = os.path.join(DATA_PROC, "x_test_dic_compiled.csv")
NEW_Y_TEST = os.path.join(DATA_CLEAN, "y_test_dic.csv")
NEW_X_TEST = os.path.join(DATA_CLEAN, "x_test_dic.csv")

# read CSV rows removing the index from B and C
def read_csv_rows(path, has_index=False, has_header=False):
    """
    Reads CSV and returns:
    - header (as list)
    - list of row tuples (stripped of index if has_index=True)
    """
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)

        header = None
        if has_header:
            header = next(reader)
        
        rows = []
        for row in reader:
            if has_index:
                row = row[1:]      # remove index column
            rows.append(tuple(row))

        return header, rows
    
def sorting(y_old_path, y_path, x_path, new_y_path, new_x_path):
    print("Reading CSV files...")
    # load all files
    header_A, rows_A = read_csv_rows(y_old_path, has_index=False, has_header=True)
    header_B, rows_B = read_csv_rows(y_path, has_index=True, has_header=True)
    header_C, rows_C = read_csv_rows(x_path, has_index=False, has_header=False)

    print("Building ordering dictionary and sorting Y...")
    # build ordering dictionary and sort B
    order_index = {row: i for i, row in enumerate(rows_A)}
    rows_B_sorted = sorted(rows_B, key=lambda r: order_index[r])

    print("Reordering X according to Y...")
    # build permutation and reorder C
    perm = [order_index[row] for row in rows_B]

    rows_C_sorted = [None] * len(rows_C)
    for old_pos, new_pos in enumerate(perm):
        rows_C_sorted[new_pos] = rows_C[old_pos]

    print("Writing sorted CSV files...")
    # save output w/o index or header
    def write_csv(path, rows, header=None):
        with open(path, "w", newline='', encoding='utf-8') as f:
            writer = csv.writer(f, lineterminator='\n')
            if header is not None:
                writer.writerow(header[1:]) # drop index header
            for row in rows:
                writer.writerow(row)

    write_csv(new_y_path, rows_B_sorted, header_B)
    write_csv(new_x_path, rows_C_sorted, header=None)

def main():
    print("===== DIC dataset sorting script started =====")

    print("Sorting train set...")
    # ---------  TRAIN SET  ----------
    sorting(Y_TRAIN_OLD, Y_TRAIN, X_TRAIN, NEW_Y_TRAIN, NEW_X_TRAIN)

    print("Sorting test set...")
    # ---------  TEST SET  ----------
    sorting(Y_TEST_OLD, Y_TEST, X_TEST, NEW_Y_TEST, NEW_X_TEST)

    print("===== DIC dataset sorting script finished =====")
    

if __name__ == "__main__":
    main()