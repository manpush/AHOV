import datetime
import datetime as dt
import math
import geopy.distance
from geopy.distance import *
from numpy import *
import plotly.express as px
import numpy as np

settings = [
    5,  # количество учитываемых ближайших источников информации
]

from sklearn.neighbors import KNeighborsRegressor


class hydrometer_data:
    def __init__(self, coordinates, x, y):
        self.CoordinatesXY = coordinates
        rad_wind = 0
        if x < 0 and y > 0:
            rad_wind = pi + math.atan(y / x)
        elif x < 0 and y < 0:
            rad_wind = pi + math.atan(y / x)
        elif x > 0 and y < 0:
            rad_wind = 3 * (pi / 2) - math.atan(y / x)
        elif x == 0 and y > 0:
            rad_wind = pi / 2
        elif x == 0 and y < 0:
            rad_wind = 3 * pi / 4
        elif y == 0 and x >= 0:
            rad_wind = 0
        elif y == 0 and x < 0:
            rad_wind = pi / 2
        else:
            rad_wind = math.atan(y / x)
        rad_wind = (pi*5/2 - rad_wind) % (2*pi)
        self.wind_strong = math.sqrt(x * x + y * y)
        self.wind_deg = math.degrees(rad_wind)
        self.wind_rad = rad_wind
        self.wind_proections = [x, y]


class hydrometer_data_calculate:
    def __init__(self, data):
        self.data = data

    def get_point_from_data(self, x, y):
        knn = KNeighborsRegressor(n_neighbors=5, weights='distance')
        X = np.array([[i.CoordinatesXY[0], i.CoordinatesXY[1]] for i in self.data])
        Y = np.array([[i.wind_proections[0], i.wind_proections[1]] for i in self.data])
        knn.fit(X, Y)
        res = knn.predict([[x, y]])
        return hydrometer_data([x, y], res[0][0], res[0][1])


class event_of_problem:  # объект аварии
    def __init__(self, lat, lot, type_agent, mass_active, time_of_start, agents_params_csv_file_path,
                 depth_of_possible_infection_zone_csv_file_path):
        self.lat = lat  # координаты центра аварии
        self.lon = lot

        self.agents_params = np.recfromcsv(agents_params_csv_file_path, delimiter=';')  # csv с параметрами разных АХОВ
        self.depth_of_possible_infection_zone_params = np.recfromcsv(depth_of_possible_infection_zone_csv_file_path,
                                                                     delimiter=';')  # csv с параметрами возможной глубины поражения
        self.type_agent = type_agent  # индекс исследуемого АХОВ
        self.mass_active = mass_active  # масса выброшеных ахов
        self.time_of_start = time_of_start  # Время проишествия

        # СВУВ - К5 - К8 - скорость переноса воздушных масс
        self.SVU_dict = {"Инверсия": [1, 0.081, lambda x: np.interp(x, [1, 2, 3, 4], [5, 10, 16, 21])],
                         "Изометрия": [0.23, 0.133,
                                       lambda x: np.interp(x, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
                                                           [6, 12, 18, 24, 29, 35, 41, 47, 53, 59, 65, 71, 76, 82,
                                                            88])],
                         "Конвекция": [0.08, 0.235, lambda x: np.interp(x, [1, 2, 3, 4], [7, 14, 21, 28])]}

    def get_SVU(self, time, type_of_weather, wind_speed):
        # time - обычное время, нужно для определения ночь/утро/день/вечер
        # type_of_weather - тип погоды: 0 - ясно или переменная обл, 1 - сплошная обл
        if (dt.datetime(hour=0, year=1, month=1, day=1) <= time < dt.datetime(hour=11, year=1, month=1, day=1)) \
                or (
                dt.datetime(hour=19, year=1, month=1, day=1) <= time < dt.datetime(hour=23, minute=59, year=1, month=1,
                                                                                   day=1)):
            if type_of_weather == 0:
                if wind_speed > 4:
                    return self.SVU_dict['Изометрия']
                else:
                    return self.SVU_dict['Инверсия']
            elif type_of_weather == 1:
                return self.SVU_dict['Изометрия']
        elif dt.time(hour=11) <= time < dt.time(hour=19):
            if type_of_weather == 0:
                if wind_speed < 2:
                    return self.SVU_dict['Конвекция']
                else:
                    return self.SVU_dict['Изометрия']
            elif type_of_weather == 1:
                return self.SVU_dict['Изометрия']

    @staticmethod
    def Qe1(K1, K3, K5, K7, Q0):
        """
        Определение эквивалентного количества АХОВ в первичном облаке (c.41)\n
        :param K1: коэффициент условий хранения (табл.1);
        :param K3: коэффициент пересчета с хлора (табл.1);
        :param K5: коэффициент учитывающий СВУВ (табл.2);
        :param K7: коэффициент температуры воздуха (табл.1);
        :param Q0: количество вещества, участвующего в аварии;
        :return: Количество АХОВ в первичном облаке;
        """
        return K1 * K3 * K5 * K7 * Q0

    @staticmethod
    def Qe2(K1, K2, K3, K4, K5, K6, K7, Q0, h, d):
        """
        Определение эквивалентного количества АХОВ во вторичном облаке. (с45)\n
        :param K1: коэффициент условий хранения (табл.1);
        :param K2: коэффициент физико-химических свойств (табл.1);
        :param K3: коэффициент пересчета с хлора (табл.1);
        :param K4: коэффициент скорости ветра (табл.4);
        :param K5: коэффициент учитывающий СВУВ (табл.2);
        :param K6: коэффициент зависящий от времени с начала аварии;
        :param K7: коэффициент температуры воздуха (табл.1);
        :param Q0: количество вещества. участвующего в аварии. т;
        :param h: толщина слоя АХОВ. м (если нет вылива в поддон принимается равной 0.05);
        :param d: плотность АХОВ. т/м3
        :return: количество АХОВ во вторичном облаке
        """
        return (1 - K1) * K2 * K3 * K4 * K5 * K6 * K7 * (Q0 / (h * d))

    @staticmethod
    def K4(V):
        """
        Значения K4 в зависимости от скорости ветра\n
        :param V: Скорость ветра
        :return: значение К4
        """
        return np.interp(V, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15],
                         [1, 1.33, 1.67, 2.0, 2.34, 2.67, 3.0, 3.34, 3.67, 4.0, 5.68])

    @staticmethod
    def K6(K2, K4, K7, h, d):
        """
        K6 определяется после нахождения времени испарения АХОВ\n
        :param K2:коэффициент физико-химических свойств (табл.1);
        :param K4:коэффициент скорости ветра (табл.4);
        :param K7:коэффициент температуры воздуха (табл.1);
        :param h:толщина слоя АХОВ. м (если нет вылива в поддон принимается равной 0.05);
        :param d:плотность АХОВ. т/м3
        :return: коэффициент зависящий от времени с начала аварии;
        """
        return (h * d) / (K2, K4, K7)

    @staticmethod
    def Sv(G, V):
        """
        :param G:глубина зоны заражения. км;
        :param V:скорость ветра
        :return:Площадь зоны возможного заражения
        """
        return 0.00872 * G * G * (Fzone(V))

    def get_full_cricle(self, lat, lot, r1, L, alpha):
        Ox, Oy = point_lat_lot(distance(kilometers=(r1*2+L)/2-r1).destination(Point(latitude=lat, longitude=lot), alpha))
        return Oy, Ox, (r1*2+L)/2, self.create_cricle_dots(Oy, Ox, (r1*2+L)/2)

    def create_cricle_dots(self, Ox, Oy, r):
        '''
            возвращает координаты точек находящихся на окружности облака
            '''
        points1 = []
        for i in range(32):
            points1.append(point_lat_lot(distance(kilometers=r).destination(Point(Ox, Oy), i*(360/32))))
        return points1

    def get_sector_of_cricle(self, lat1, lot1, r1, beta, L, alpha):
        """
        возвращает точки облака радиустом r1 из точки lat-lot, угол распространиения beta, глубиной L, углом ветра alpha
        :param lat1: координата первичного облака
        :param lot1: координата первичного облака
        :param r1: радиус облака
        :param beta: угол распространения веществ
        :param L: глубина распространения
        :param alpha: угол направления ветра
        :return: массив координат обозначающих границу облака
        """
        # координаты и радиус большей окружности
        lot2, lat2 = point_lat_lot(distance(kilometers=L).destination(Point(lat1, lot1), alpha))
        r2 = L * math.sin(math.radians(beta) / 2) + r1

        # углы
        f1 = (180 + beta) / 32
        f2 = (180 - beta) / 32

        resarr = get_dots(f1, f2, (lat1, lot1, r1), (lat2, lot2, r2), [270+alpha-beta/2, 90+alpha+beta/2])

        return (lat2, lot2, r2, resarr)  # [[t1x, t1y],[t2x, t2y],[t4x, t4y],[t3x, t3y]]#

    def get_traps(self, hdc, time, radius_of_first_cloud, type_of_weater, mass_active, start_data, lat=None, lot=None):
        """
        массив массивов точек облаков глубиной 1км и менее
        :param lat:
        :param lot:
        :param hdc:
        :param time:
        :param radius_of_first_cloud:
        :param type_of_weater:
        :param start_data:
        :return:
        """
        if lat is None: lat = self.lat
        if lot is None: lot = self.lon

        wind = hdc.get_point_from_data(lat, lot)  # скорость ветра в конкретной точке
        if radius_of_first_cloud == 0:
            radius_of_first_cloud = 1
        # вычисляем глубину распространения
        Gv = get_Gv(mass_active, wind.wind_strong, self.depth_of_possible_infection_zone_params)
        Gp = (time - self.time_of_start).seconds / 3600 * self.get_SVU(time, type_of_weater, wind.wind_strong)[2](
            wind.wind_strong)
        if Gv < Gp:
            G = Gv
        else:
            G = Gp
        # если глубина больше 1км, то считаем на 1 км
        if G > 1:
            G = 1
        # получить один отрезок облака глубиной в километр
        if (wind.wind_strong < 0.5):
            # если ветер слабый то рисуем только окружность
            lat, lot, radius_of_first_cloud, data = self.get_full_cricle(lat, lot, radius_of_first_cloud, G, wind.wind_deg)
            start_data.append(data)
        else:
            # иначе рисуем раздутое облако
            lat, lot, radius_of_first_cloud, data = self.get_sector_of_cricle(lat, lot, radius_of_first_cloud,
                                                                              Fzone(wind.wind_strong),
                                                                              G, wind.wind_deg)
            start_data.append(data)
        #return start_data
        if G < 1:
            # если это последнее звено то возвращаем полученное
            return start_data
        else:
            # если есть продолжение то продолжаем с учётом осевшего вещества и прошедшего времени
            mass_active = mass_active - mass_active / min(Gv, Gp)
            time_on_G = G / self.get_SVU(time, type_of_weater, wind.wind_strong)[2](wind.wind_strong)
            return self.get_traps(hdc, time - datetime.timedelta(minutes=int(time_on_G * 60)),
                           radius_of_first_cloud, type_of_weater, mass_active, start_data,lat, lot)


def Fzone(n):
    return np.interp(n, [0.5, 1, 2, 100],
                     [179.99, 90, 45, 45])


def get_Gv(mass, v, depth_of_possible_infection_zone_params):
    X = [depth_of_possible_infection_zone_params[0][i] for i in range(1,len(depth_of_possible_infection_zone_params[0]))]
    Y = [depth_of_possible_infection_zone_params[i][0] for i in range(len(depth_of_possible_infection_zone_params))]
    grid = [[depth_of_possible_infection_zone_params[i][j] for j in
             range(1, len(depth_of_possible_infection_zone_params[i]))] for i in
            range(1, len(depth_of_possible_infection_zone_params))]
    # Find the vales that x and y are between.
    xi, yi = None, None
    for i, (x1, x2) in enumerate(zip(X[:-1], X[1:])):
        if x1 <= mass <= x2:
            xi, w_x2, w_x1 = i, (mass - x1) / (x2 - x1), (x2 - mass) / (x2 - x1)
            break
    for i, (y1, y2) in enumerate(zip(Y[:-1], Y[1:])):
        if y1 <= v <= y2:
            yi, w_y2, w_y1 = i - 1, (v - y1) / (y2 - y1), (y2 - v) / (y2 - y1)
            break
    if xi is None or yi is None:
        return False
        # You could add special cases to interpolate past the range if you would like.
    # Find the weighted average between the four corners.
    ave = grid[yi][xi] * w_y1 * w_x1
    ave += grid[yi][xi + 1] * w_y1 * w_x2
    ave += grid[yi + 1][xi] * w_y2 * w_x1
    ave += grid[yi + 1][xi + 1] * w_y2 * w_x2
    return ave

def point_lat_lot(point):
    return [point.longitude, point.latitude]

def get_dots(f1, f2, o1, o2, start_angles):
    '''
    возвращает координаты точек находящихся на внешних секторах окружностей первичного и вторичного облака
    :param t1x:
    :param t1y:
    :param t4x:
    :param t4y:
    :param f1:
    :param f2:
    :param o1:
    :param o2:
    :return:
    '''
    points1 = []
    points2 = []
    for i in range(32):
        points1.append(point_lat_lot(distance(kilometers=o1[2]).destination(Point(o1[0], o1[1]), -i*f2 + start_angles[0])))
        points2.append(point_lat_lot(distance(kilometers=o2[2]).destination(Point(o2[0], o2[1]), -i*f1 + start_angles[1])))
    return points1 + points2


# от сюда начинается основной код
# создание массива метеовышек (широта долгота, проекция скорости ветра по х, проекция по у)
hdc = hydrometer_data_calculate(
    [
        hydrometer_data([54.8, 47.9], 1, 1),
        hydrometer_data([55.8, 47.9], 1, 0),
        hydrometer_data([56.8, 47.9], 0, 1),
        hydrometer_data([54.8, 48.9], 0, -1),
        hydrometer_data([55.8, 48.9], -1, -1),
        hydrometer_data([56.8, 48.9], 0, 1),
        hydrometer_data([54.8, 49.9], 1, -1),
        hydrometer_data([55.8, 49.9], -1, 0),
        hydrometer_data([56.8, 49.9], -1, 0),

    ]
)
# создание объекта аварии
ep = event_of_problem(55.871917, 48.986879, 4, 1000, datetime.datetime(hour=0, year=1, month=1, day=1), 'nir/coeff_set.csv',
                      'nir/depth_wind.csv')
# создание массива для отображения метеостанций на карте (красные точки)
hdcres = np.array(())
for i in hdc.data:
    hdcres = np.append(hdcres, (i.CoordinatesXY[0], i.CoordinatesXY[1], i.wind_strong))
hdcres = np.reshape(hdcres, (-1, 3))

# отметить на карте метеоточки
fig = px.scatter_mapbox(lat=[x[0] for x in hdcres],
                        lon=[x[1] for x in hdcres],
                        color_discrete_sequence=["red"], height=700)
res = ep.get_traps(hdc, datetime.datetime(hour=2, year=1, month=1, day=1), 0, 0,
                                         ep.mass_active, [])
fig.update_layout(
    mapbox = {
        'style': "open-street-map",
        'zoom': 12, 'layers': [{
            'source': {
                'type': "FeatureCollection",
                'features': [{
                    'type': "Feature",
                    'geometry': {
                        'type': "MultiPolygon",
                        'coordinates': [[
                            i
                        ]]
                    }
                } for i in res]
            },
            'type': "fill", 'below': "traces", 'color': "yellow"}]},
    margin = {'l':0, 'r':0, 'b':0, 't':0})



fig.show()
