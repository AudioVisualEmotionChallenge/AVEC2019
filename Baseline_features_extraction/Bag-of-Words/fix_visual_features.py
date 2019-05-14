import csv
from os import listdir
from os.path import basename, join

VISUAL_FEATURES_FOLDER = '../visual_features/'  # should only contain openface feature csvs
VISUAL_FEAUTRES_FIXED = '../visual_features_fixed/'


def fix_single_csv(input_path, output_path):
    with open(input_path, 'r') as csv_in:
        reader = csv.reader(csv_in, delimiter=',')
        with open(output_path, 'w') as csv_out:
            header = next(reader)
            header[0] = 'name'
            writer = csv.writer(csv_out, delimiter=';')
            writer.writerow(header)
            for line in reader:
                line[0] = basename(input_path)
                writer.writerow(line)


def fix_folder(input_folder, output_folder):
    for f in listdir(input_folder):
        print(f)
        fix_single_csv(join(input_folder, f), join(output_folder, f))


if __name__ == '__main__':
    fix_folder(VISUAL_FEATURES_FOLDER, VISUAL_FEAUTRES_FIXED)
