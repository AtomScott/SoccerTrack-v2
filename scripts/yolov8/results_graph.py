import numpy as np
import json
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
    parser.add_argument('--input_10', type=str, required=True, help='Input dataset_10.json file for making graphs') # ...\outputs\results.json\dataset_10.json (file path)
    parser.add_argument('--input_50', type=str, required=True, help='Input dataset_50.json file for making graphs') # ...\outputs\results.json\dataset_50.json (file path)
    parser.add_argument('--input_100', type=str, required=True, help='Input dataset_100.json file for making graphs') # ...\outputs\results.json\dataset_100.json (file path)
    parser.add_argument('--output_all_directory', type=str, required=True, help='Output directory for saving all class graphs') # ...\outputs\graph\all_class (directory path)
    parser.add_argument('--output_person_directory', type=str, required=True, help='Output directory for saving person class graphs') # outputs\graph\person_class (directory path)
    parser.add_argument('--output_ball_directory', type=str, required=True, help='Output directory for saving ball class graphs') # outputs\graph\ball_class (directory path)
    return parser.parse_args()


def main():

    args = parse_args()
    
    # dict in json to np
    all_class_np = np.full([3,4], 1.0)
    person_class_np = np.full([3,4], 1.0)
    ball_class_np = np.full([3,4], 1.0)

    for i,dataset in enumerate([10, 50, 100]):

        exec("json_open = open(args.input_" + str(dataset) + ", 'r')")
        results_dict = json.load(eval('json_open'))
        print(results_dict)

        all_class_data_dict = results_dict["all"]
        person_class_data_dict = results_dict["person"]
        ball_class_data_dict = results_dict["ball"]

        all_class_np[i] = list(all_class_data_dict.values())
        person_class_np[i] = list(person_class_data_dict.values())
        ball_class_np[i] = list(ball_class_data_dict.values())


    # name list of validation indexes
    index_name_list = ['Precision', 'Recall', 'mAP50', 'mAP50-95']
    
    # make results graph from all class np
    for i in range(4):
        x = [10, 50, 100]
        y = all_class_np[:, i]
        plt.title(index_name_list[i]) 
        plt.xlabel("train dataset") 
        plt.ylabel(index_name_list[i]) 
        plt.plot(x, y, color ="red")
        file_path = os.path.join(args.output_all_directory, str(index_name_list[i]) + ".png")
        plt.savefig(file_path)

    # make results graph from person class np
    for i in range(4):
        x = [10, 50, 100]
        y = person_class_np[:, i]
        plt.title(index_name_list[i]) 
        plt.xlabel("train dataset") 
        plt.ylabel(index_name_list[i]) 
        plt.plot(x, y, color ="red")
        file_path = os.path.join(args.output_person_directory, str(index_name_list[i]) + ".png")
        plt.savefig(file_path)

    # make results graph from ball class np
    for i in range(4):
        x = [10, 50, 100]
        y = ball_class_np[:, i]
        plt.title(index_name_list[i]) 
        plt.xlabel("train dataset") 
        plt.ylabel(index_name_list[i]) 
        plt.plot(x, y, color ="red")
        file_path = os.path.join(args.output_ball_directory, str(index_name_list[i]) + ".png")
        plt.savefig(file_path)


if __name__ == "__main__":
    main()