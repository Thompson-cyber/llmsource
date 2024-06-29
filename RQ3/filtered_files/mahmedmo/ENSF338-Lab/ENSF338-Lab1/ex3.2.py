import json
import timeit
import numpy as np
import matplotlib.pyplot as plt

# update_all_sizes generated with ChatGPT
def update_all_sizes(obj):
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key == 'size':
                obj[key] = 35
            else:
                update_all_sizes(value)
    elif isinstance(obj, list):
        for item in obj:
            update_all_sizes(item)

def test_update_all_sizes(json_data,run_no):
    elapsed_time = timeit.timeit(lambda: update_all_sizes(json_data), number=run_no)
    print(f'Average elapsed time: {elapsed_time/run_no} seconds')

def update_json(file_in, file_out):
    with open(file_in, 'r') as j_file:
        json_data = json.load(j_file)
    print("Exercise 3.2 Linear Regression...")
    record_counts = [1000, 2000, 5000, 10000]
    avgtimes = []
    for count in record_counts:
        sliced_data = json_data[:count]

        time_taken = timeit.timeit(lambda: update_all_sizes(sliced_data), number=100) / 100
        avgtimes.append(time_taken)
        print(f"Average time to process first {count} records: {time_taken:.6f} seconds")
    
    json_data.reverse()
    with open(file_out, 'w') as j_file:
        json.dump(json_data, j_file,indent = 4)

    slope, intercept = np.polyfit(record_counts, avgtimes, 1)
    linevalues = [slope * x + intercept for x in record_counts]
    # linear regression plotted via ChatGPT
    plt.scatter(record_counts, avgtimes)
    plt.plot(record_counts, linevalues, 'r')
    plt.xlabel('Number of Records')
    plt.ylabel('Average Processing Time (seconds)')
    plt.title('Processing Time vs Number of Records')
    plt.savefig('ENSF338-Lab1/output.3.2.png')
    plt.close()
    print("The linear model is: t = %.2e * n + %.2e" % (slope, intercept))

    print("Linear Regression complete! Saved to output.3.2.png")


file_in = 'ENSF338-Lab1/large-file.json'
file_out = 'ENSF338-Lab1/output.2.3.json'

update_json(file_in, file_out)
