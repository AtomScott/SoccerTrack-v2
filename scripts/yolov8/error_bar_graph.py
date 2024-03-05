import numpy as np
import matplotlib.pyplot as plt
import argparse
import os


def parse_args():
    """
    Parse command line arguments.

    Returns:
        Namespace: The arguments namespace.
    """
    parser = argparse.ArgumentParser(description='Save the graph on a specific path.')
    parser.add_argument('--input_randomstate1', type=str, required=True, help='Input all_results_np1.npy file for making a error bar graph') # ...\outputs\graph\randomstate1\all_results_np1.npy (file path)
    parser.add_argument('--input_randomstate2', type=str, required=True, help='Input all_results_np2.npy file for making a error bar graph') # ...\outputs\graph\randomstate1\all_results_np2.npy (file path)
    parser.add_argument('--input_randomstate3', type=str, required=True, help='Input all_results_np3.npy file for making a error bar graph') # ...\outputs\graph\randomstate1\all_results_np3.npy (file path)
    parser.add_argument('--input_randomstate4', type=str, required=True, help='Input all_results_np4.npy file for making a error bar graph') # ...\outputs\graph\randomstate1\all_results_np4.npy (file path)
    parser.add_argument('--input_randomstate5', type=str, required=True, help='Input all_results_np5.npy file for making a error bar graph') # ...\outputs\graph\randomstate1\all_results_np5.npy (file path)
    parser.add_argument('--output_all_directory', type=str, required=True, help='Output directory for saving all class graphs') # outputs\graph\randomstate1\all_class (directory path)
    parser.add_argument('--output_person_directory', type=str, required=True, help='Output directory for saving person class graphs') # outputs\graph\randomstate1\person_class (directory path)
    parser.add_argument('--output_ball_directory', type=str, required=True, help='Output directory for saving ball class graphs') # outputs\graph\randomstate1\ball_class (directory path)
    return parser.parse_args()


def main():

    args = parse_args()

    # load a npy file of each randomstate
    for i in range(1,6):
        exec("np_open = args.input_randomstate" + str(i))
        exec("randomstate_np" + str(i) + "= np.load(eval('np_open'))")


    # calculate means of each indexes of each class
    
    # name list of validation indexes
    index_name_list = ['Precision', 'Recall', 'mAP50', 'mAP50-95']

    # name list of classes
    class_name_list = ['all', 'person', 'ball']

    # mean_np have means of each validation index of each trainset of each class
    mean_np = np.full((3,3,4), 0.0)
    # sum_np have sums of each validation index of each trainset of each class
    sum_np = np.full((3,3,4), 0.0)
    # sd_np have standard deviations of each validation index of each trainset of each class
    sd_np = np.full((3,3,4), 1.0)
    # value_np have values of each validation index of each trainset of each class
    value_np = np.full(5, 1.0)


    for i in range(4):
        for j in range(3):
            for k in range(3):
                for l, randomstate in enumerate(range(1,6)):
                    exec("value_np[l] = randomstate_np" + str(randomstate) + "[k,j,i]")
                mean_np[k,j,i] = np.mean(eval("value_np"))
                sd_np[k,j,i] = np.std(eval("value_np"))


    # make error bar graphs
    for k in range(3):
        for i in range(4):
            x = [10, 50, 100]
            y = [mean_np[k,j,i] for j in range(3)]
            y_err = [sd_np[k,j,i] for j in range(3)]
            plt.figure()
            plt.title(index_name_list[i]) 
            plt.xlabel("train dataset") 
            plt.ylabel(index_name_list[i]) 
            plt.errorbar(x, y, yerr = y_err, capsize=5, markersize=10, ecolor = 'black', markeredgecolor = "black", color = "black")
            plt.ylim(0, 1)
            exec("class_directory = args.output_" + str(class_name_list[k]) + "_directory")
            file_path = os.path.join(eval("class_directory"), str(index_name_list[i]) + ".png")
            plt.savefig(file_path)


if __name__ == "__main__":
    main()