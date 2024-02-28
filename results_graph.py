import numpy as np
import json
import matplotlib.pyplot as plt

def main():

    # dict in json to np
    results_np = np.full([3,5], 1)

    for index,dataset in enumerate([10, 50, 100]):

        json_open = open('results_json\det_metrics_dict_' + str(dataset) + '.json', 'r')
        json_load = json.load(json_open)

        for v in json_load.values():
            results_np[index,:] = v

        break

    print(results_np)

    # make results graph from np
    x = [10, 50, 100]
    y = results_np[:, 3]
    plt.title("Line graph") 
    plt.xlabel("train dataset") 
    plt.ylabel("mAP50-95") 
    plt.plot(x, y, color ="red") 
    plt.show()


if __name__ == "__main__":
    main()