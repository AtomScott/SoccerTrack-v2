import numpy as np
import json
import matplotlib.pyplot as plt

def main():

    # dict in json to np
    all_class_np = np.full([3,4], 1.0)
    person_class_np = np.full([3,4], 1.0)
    ball_class_np = np.full([3,4], 1.0)

    for i,dataset in enumerate([10, 50, 100]):

        json_open = open('C:\\Users\\黒田堅仁\\OneDrive\\GitHub\\SoccerTrack-v2\\outputs\\results_json\\dataset_' + str(dataset) + '.json', 'r')
        results_dict = json.load(json_open)

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
        plt.show()
        plt.savefig("C:\\Users\\黒田堅仁\\OneDrive\\GitHub\\SoccerTrack-v2\\outputs\\graph\\all_class\\" + index_name_list[i] + ".json")

    # make results graph from person class np
    for i in range(4):
        x = [10, 50, 100]
        y = person_class_np[:, i]
        plt.title(index_name_list[i]) 
        plt.xlabel("train dataset") 
        plt.ylabel(index_name_list[i]) 
        plt.plot(x, y, color ="red") 
        plt.show()
        plt.savefig("C:\\Users\\黒田堅仁\\OneDrive\\GitHub\\SoccerTrack-v2\\outputs\\graph\\person_class\\" + index_name_list[i] + ".json")

    # make results graph from ball class np
    for i in range(4):
        x = [10, 50, 100]
        y = ball_class_np[:, i]
        plt.title(index_name_list[i]) 
        plt.xlabel("train dataset") 
        plt.ylabel(index_name_list[i]) 
        plt.plot(x, y, color ="red") 
        plt.show()
        plt.savefig("C:\\Users\\黒田堅仁\\OneDrive\\GitHub\\SoccerTrack-v2\\outputs\\graph\\ball_class\\" + index_name_list[i] + ".json")


if __name__ == "__main__":
    main()