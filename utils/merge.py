import glob
import pandas as pd
import os
import cv2


def merge_report(file_list, dir):
    # filenames
    excel_names = []

    for file in file_list:
        try:
            # This prevents parsing corrupt, empty or unreadable sheets
            fNames = pd.read_excel(file, engine="openpyxl", sheet_name="Report")
            excel_names.append(file)
        except Exception as e:
            print("Exception for file: " + file + ", Error: " + str(e))

    # print(excel_names)
    # read them in
    excels = [pd.ExcelFile(name) for name in excel_names]

    # turn them into dataframes
    frames = [x.parse(x.sheet_names[0], header=None, index_col=None) for x in excels]

    # delete the first row for all frames except the first
    # i.e. remove the header row -- assumes it's the first
    frames[1:] = [df[1:] for df in frames[1:]]

    # concatenate them..
    combined = pd.concat(frames)

    # write it out
    combined.to_excel(
        dir + "/Master_Report.xlsx",
        header=False,
        index=False,
        sheet_name="Verification",
    )


def concat_images(image_paths, dir):
    images = []
    for image_path in image_paths:
        images.append(cv2.imread(image_path))
        print('Concatenating for :'+image_path)
    im_h = cv2.hconcat(images)
    cv2.imwrite(os.path.join(dir, "proj_concat.png"), im_h)


expt_dir = "/nas/vista-hdd01/users/raidurga/outputs/ai2ai_run_004/expts"
img_dirs = [
    "img_001",
    "img_002",
    "img_003",
    "img_004",
    "img_005",
    "img_006",
    "img_007",
    "img_008",
    "img_009",
    "img_0010",
]

merge_reports = True
concat_images_h = False


if __name__ == "__main__":

    # if merge_reports:
    #     for imgdir in img_dirs:
    #         file_list = glob.glob(expt_dir + imgdir + "/*/master-report.xlsx")
    #         merge_report(file_list, os.path.join(expt_dir, imgdir))
    #         print("Merge completed for - " + imgdir)
    if merge_reports:
        file_list = glob.glob(expt_dir + "/*/master-report.xlsx")
        merge_report(file_list, expt_dir)
        print("Merge completed for - " + expt_dir)

    if concat_images_h:
        for imgdir in img_dirs:
            image_paths = glob.glob(expt_dir + imgdir + "/*/projected.png")
            image_paths.sort()
            concat_images(image_paths, os.path.join(expt_dir, imgdir))
