from preprocess_for_test import preprocess, postprocess
from TestBRBS_care import run_test
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

if __name__ == "__main__":
    # input_dir = '/input/'
    # output_dir = '/output/LiSeg_pred/'
    input_dir = '../1009/'
    output_dir = '../output/LiSeg_pred/'
    processed_dir = '../processed1009/'
    pred_dir = './results_care_256_256_32_mi/BRBS_care_all/seg/'

    preprocess(input_dir, processed_dir)
    run_test(processed_dir)
    postprocess(input_dir, pred_dir, output_dir)
    