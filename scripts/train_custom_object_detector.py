# import the necessary packages
import multiprocessing
import argparse
import dlib

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument(
    "-t", "--training", required=True, help="path to input training XML file"
)
ap.add_argument(
    "-m", "--model", required=True, help="path serialized dlib object predictor model"
)
args = vars(ap.parse_args())

# grab the default options for dlib's object predictor
print("[INFO] setting object predictor options...")
options = dlib.simple_object_detector_training_options()

# Since faces are left/right symmetric we can tell the trainer to train a
# symmetric detector.  This helps it get the most value out of the training
# data.
options.add_left_right_image_flips = False

# The trainer is a kind of support vector machine and therefore has the usual
# SVM C parameter.  In general, a bigger C encourages it to fit the training
# data better but might lead to overfitting.  You must find the best C value
# empirically by checking how well the trained detector works on a test set of
# images you haven't trained on.  Don't just leave the value set at 5.  Try a
# few different C values and see what works best for your data.
options.C = 5

# tell the dlib object predictor to be verbose and print out status
# messages our model trains
options.be_verbose = True

options.upsample_limit = 0

# number of threads/CPU cores to be used when training -- we default
# this value to the number of available cores on the system, but you
# can supply an integer value here if you would like
options.num_threads = multiprocessing.cpu_count() // 2

# log our training options to the terminal
print("[INFO] object predictor options:")
print(options)
# train the object predictor
print("[INFO] training object predictor...")
dlib.train_simple_object_detector(args["training"], args["model"], options)
