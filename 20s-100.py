

import csv
from collections import defaultdict
from random import uniform
from math import sqrt

import dash
import dash_core_components as dcc
import dash_html_components as html

from matplotlib import colors as mcolors
import matplotlib.pyplot as plt
import mpld3
from mpld3._server import serve

dataToConvert = []


def csv_dict_writer(path, fieldnames, data):
    
    with open(path, "w", newline='') as out_file:
        writer = csv.DictWriter(out_file, delimiter=',', fieldnames=fieldnames)
        writer.writeheader()
        for row in data:
            writer.writerow(row)


def csv_reader(path):
    with open(path, "r", newline="") as file:
        # читаем файл целиком
        reader = csv.reader(file)
        
        str1 = ['localhost CPU']
        str2 = ['localhost Memory']
        str3 = ['localhost Network I/O']
        count = 0
        temp = []
        for row in reader:
            cur_arr = row[1].split(',')
            if (cur_arr == str1 or cur_arr == str2 or cur_arr == str3):
                count = count + 1
                temp.append(row[0])
                if (count == 3):
                    dataToConvert.append(list(map(int, temp)))
                    temp = []
                    count = 0

def point_avg(points):
    
    dimensions = len(points[0])

    new_center = []

    for dimension in range(dimensions):
        dim_sum = 0  # dimension sum
        for p in points:
            dim_sum += p[dimension]

        # average of each dimension
        new_center.append(dim_sum / float(len(points)))

    return new_center


def update_centers(data_set, assignments):
    
    new_means = defaultdict(list)
    centers = []
    for assignment, point in zip(assignments, data_set):
        new_means[assignment].append(point)

    for points in new_means.values():
        centers.append(point_avg(points))

    return centers


def assign_points(data_points, centers):
   
    assignments = []
    for point in data_points:
        shortest = float('inf')  # positive infinity
        shortest_index = 0
        for i in range(len(centers)):
            val = distance(point, centers[i])
            if (val < shortest):
                shortest = val
                shortest_index = i
        assignments.append(shortest_index)
    return assignments


def distance(a, b):
    
    dimensions = len(a)

    _sum = 0
    for dimension in range(dimensions):
        difference_sq = (a[dimension] - b[dimension]) ** 2
        _sum += difference_sq
    return sqrt(_sum)


def generate_k(data_set, k):
    
    centers = []
    dimensions = len(data_set[0])
    min_max = defaultdict(int)

    for point in data_set:
        for i in range(dimensions):
            val = point[i]
            min_key = 'min_%d' % i
            max_key = 'max_%d' % i
            if (min_key not in min_max or val < min_max[min_key]):
                min_max[min_key] = val
            if (max_key not in min_max or val > min_max[max_key]):
                min_max[max_key] = val

    for _k in range(k):
        rand_point = []
        for i in range(dimensions):
            min_val = min_max['min_%d' % i]
            max_val = min_max['max_%d' % i]

            rand_point.append(uniform(min_val, max_val))

        centers.append(rand_point)

    return centers


def k_means(dataset, k ):
    print('grup: ', k)
    global k_means_res
    k_points = generate_k(dataset, k)
    assignments = assign_points(dataset, k_points)
    old_assignments = None
    while assignments != old_assignments:
        new_centers = update_centers(dataset, assignments)
        old_assignments = assignments
        assignments = assign_points(dataset, new_centers)
        k_means_res = list(zip(assignments, dataset))
    print('centers: ', new_centers)



    wss = 0
    arr_size_y = len(dataset)
    arr_size_x = len(dataset[0])
    for index in range(arr_size_y):
        for index_count in range(arr_size_x):
            wss += ((k_means_res[index][1][index_count] - new_centers[k_means_res[index][0]][index_count]) ** 2)
            
    print('wss :', wss)
    sums = list(range(arr_size_x))
    totss = 0
    for index_count in range(arr_size_x):
        for index in range(arr_size_y):
            sums[index_count] += dataset[index][index_count]
        sums[index_count] = sums[index_count] / arr_size_y
        for index in range(arr_size_y):
            totss += (dataset[index][index_count] - sums[index_count])**2
    print('totss = ', totss)
    r_square = 1 - (wss * (arr_size_y - 1)) / (totss * (arr_size_y - 2))
    return r_square




csv_reader('0.2min-100-server.csv')
print("------------------------")
clast_count = 2


k_means_res = ([],[])
clast_old = k_means(dataToConvert, clast_count)
print('r_square :', clast_old)
print('k_means_res: ', k_means_res)
while(True):
    print('-------------------------------------------')
    clast_count = clast_count + 1
    clast_new = k_means(dataToConvert, clast_count)

    print('k_means_res: ', k_means_res)
    print('grup', clast_count - 1, clast_old, ' - ', 'grup', clast_count, clast_new, ' = ', (clast_old - clast_new))

    if((clast_old - clast_new) < 0.05 and (clast_old - clast_new) > -0.05):
        break
    clast_old = clast_new
    print('-------------------------------------------')


cpu = list(range(len(dataToConvert)))
disk = list(range(len(dataToConvert)))
network = list(range(len(dataToConvert)))


for index in range(len(dataToConvert)):

    cpu[index] = dataToConvert[index][0]
    disk[index] = dataToConvert[index][1]
    network[index] = dataToConvert[index][2]





colors = list(mcolors.CSS4_COLORS)

def update_graph3(selected_drop_downs):
    traces = []
    for fund in range(len(selected_drop_downs)):
        trace = {'x':[fund, fund, fund] ,'y': selected_drop_downs[fund][1], 'mode': 'markers', 'marker': {'color': colors[selected_drop_downs[fund][0] + 20], 'size': 10}, 'name': fund}
        traces.append(trace)
    return traces

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div(children=[
	html.Div([
    html.H1('Звіт з результатами про тестування навантаження'),
    html.Div([
        html.P('Час навантаження: 0.3 хвилина'),
        html.P('Кількість користувачів: 100'),
        html.P('1 кластер(синій колір) має великі скачки')

    ])
]),
    html.Div(
        dcc.Graph(
            id='graph',
            figure={
                'data': [
                    {'x': list(range(len(dataToConvert))), 'y': cpu, 'type': 'line', 'marker': {'size': 10}, 'name': 'CPU'},
                    {'x': list(range(len(dataToConvert))), 'y': disk, 'type': 'line', 'marker': {'size': 10}, 'name': 'Memory'},
                    {'x': list(range(len(dataToConvert))), 'y': network, 'type': 'line', 'marker': {'size': 10},'name': 'Network'},
                ],
                'layout': {
                    'title': 'Графік показників веб-серверу'
                }
            }
        )
    ),
    html.Div(
        dcc.Graph(
            id='graph2',
            figure={
                'data':
                    update_graph3(k_means_res),
                'layout': {
                    'title': 'Графік кластерів'
                }
            }
        )
    )
])



if __name__ == '__main__':
    app.run_server(debug=True)
