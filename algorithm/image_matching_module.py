import os
import time
import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np
import imutils
import mahotas
from scipy.spatial import distance as dist


class Data:
    def __init__(self, filename, kind) -> None:
        self._filename = filename
        self._image = cv.imread(os.path.join(kind, filename))
        self._image = cv.cvtColor(self._image, cv.COLOR_BGR2RGB)
        self._scaled = cv.resize(self._image, (1024, 1024), interpolation=cv.INTER_AREA)
        self._grayscale = cv.cvtColor(self._scaled, cv.COLOR_BGR2GRAY)
        self._matrix = cv.resize(self._grayscale, (8, 8), interpolation=cv.INTER_AREA)
        self._histogramMatrix = cv.calcHist(
            [self._grayscale], [0], None, [16], [0, 256]
        )
        self._histogram_by_medium = {}
        self._mean_brightness = np.mean(self._matrix)
        self._descriptor = []
        self._key_points = []
        self._kind = kind

    def __str__(self) -> str:
        return f"Image path: {self._filename}"

    @property
    def image(self):
        return self._image

    @property
    def path(self):
        return self._filename

    @property
    def scaled(self):
        return self._scaled

    @property
    def grayscale(self):
        return self._grayscale

    @property
    def matrix(self):
        return self._matrix

    @property
    def descriptor(self):
        return self._descriptor

    @property
    def key_points(self):
        return self._key_points

    @property
    def kind(self):
        return self._kind

    @property
    def histogram_by_medium(self):
        return self._histogram_by_medium

    @histogram_by_medium.setter
    def histogram_by_medium(self, value):
        self._histogram_by_medium = value
        self._epsilon = abs(
            self._histogram_by_medium["below_threshold"]
            - self._histogram_by_medium["above_threshold"]
        )


class DataSet:
    def __init__(self, filename_list, kind) -> None:
        self._data_list = []
        self._division_point = 0
        self._kind = kind

        for image_name in filename_list:
            self._data_list.append(Data(image_name, kind))

    def __str__(self) -> str:
        output = ""
        for data in self._data_list:
            output = output + data.path + "\n"
        return output

    @property
    def kind(self):
        return self._kind

    @property
    def images(self):
        imgs = []
        for data in self._data_list:
            imgs.append(data)
        self._images = imgs
        return self._images

    @property
    def data_list(self):
        return self._data_list

    @property
    def matrices(self):
        matrices = []
        for data in self._data_list:
            matrices.append(data.matrix)
        self._matrices = matrices
        return self._matrices


class GroupAssigner:
    def __init__(self, personal_image_list, others_image_list, epsilon) -> None:
        self._personal_image_list = personal_image_list
        self._others_image_list = others_image_list
        self._epsilon = epsilon
        self._g1 = []
        self._g2 = []
        self._g3 = []

    @property
    def g1(self):
        return self._g1

    @property
    def g2(self):
        return self._g2

    @property
    def g3(self):
        return self._g3

    def count_sum_of_matrices(self, matrices_list) -> int:
        sum = 0
        for matrix in matrices_list:
            sum += np.matrix(matrix).sum()
        return sum

    def count_mean(self):
        sum_personal = self.count_sum_of_matrices(self._personal_image_list.matrices)
        sum_others = self.count_sum_of_matrices(self._others_image_list.matrices)

        total_sum = sum_personal + sum_others
        count = (len(self._personal_image_list.data_list) * 64) + (
            len(self._others_image_list.data_list) * 64
        )
        self.medium = total_sum / count

    def generate_histogram_by_medium(self, image_list):
        for image in image_list:
            img = image.matrix
            below_threshold = 0
            above_threshold = 0

            for line in img:
                for pixel in line:
                    if pixel <= self.medium:
                        below_threshold += 1
                    else:
                        above_threshold += 1
            hist_by_medium = {}
            hist_by_medium["below_threshold"] = below_threshold
            hist_by_medium["above_threshold"] = above_threshold
            image.histogram_by_medium = hist_by_medium

    def assign_groups_1_2(self):
        for image in self._personal_image_list.data_list:
            if (
                image._histogram_by_medium["below_threshold"]
                >= image._histogram_by_medium["above_threshold"]
            ):
                self._g1.append(image)
            else:
                self._g2.append(image)

        for image in self._others_image_list.data_list:
            if (
                image._histogram_by_medium["below_threshold"]
                >= image._histogram_by_medium["above_threshold"]
            ):
                self._g1.append(image)
            else:
                self._g2.append(image)

    def assing_group_3(self):
        i = len(self._g1) - 1
        while i < 0:
            if self._g1[i]._epsilon <= self._epsilon:
                self._g3.append(self._g2[i])
                self._g2.remove(self._g2[i])
            i -= 1

        i = len(self._g2) - 1
        while i > 0:
            if self._g2[i]._epsilon <= self._epsilon:
                self._g3.append(self._g2[i])
                self._g2.remove(self._g2[i])
            i -= 1

    def run(self):
        self.count_mean()
        self.generate_histogram_by_medium(self._personal_image_list.data_list)
        self.generate_histogram_by_medium(self._others_image_list.data_list)
        self.assign_groups_1_2()
        self.assing_group_3()


class ZernikeMomentDescriptorGenerator:
    def __init__(self, image_list_personal, image_list_others):

        self.image_list_personal = image_list_personal
        self.image_list_others = image_list_others

    def run(self):
        print("\nGenerowanie cech lokalnych obrazów")

        for image_data in self.image_list_personal.data_list:
            descriptor = self.generate_image_descriptor(image_data.scaled)
            image_data._descriptor = descriptor

        for image_data in self.image_list_others.data_list:
            descriptor = self.generate_image_descriptor(image_data.scaled)
            image_data._descriptor = descriptor

        return self.image_list_personal, self.image_list_others

    def generate_image_descriptor(self, image):
        image = cv.copyMakeBorder(image, 15, 15, 15, 15, cv.BORDER_CONSTANT, value=255)
        thresh = cv.bitwise_not(image)
        thresh[thresh > 0] = 255
        edged = cv.Canny(thresh, 30, 200)

        outline = np.zeros(edged.shape, dtype="uint8")
        cnts = cv.findContours(edged, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        cnts = imutils.grab_contours(cnts)
        cnts = sorted(cnts, key=cv.contourArea, reverse=True)[0]
        cv.drawContours(outline, [cnts], -1, 255, -1)

        moments = mahotas.features.zernike_moments(outline, 21)
        return moments


class ZenrickeMomentImageMatcher:
    def __init__(self, g1, g2, g3) -> None:
        self._g1 = g1
        self._g2 = g2
        self._g3 = g3
        self._matches = []

    def run(self):
        self.compare_image_list(self._g1)
        self.compare_image_list(self._g2)
        self.compare_image_list(self._g3)

        return self._matches

    def assign_groups_for_comparision(self, group):
        list_personal = []
        list_others = []

        for image in group:
            if image.kind == "personal":
                list_personal.append(image)
            if image.kind == "others":
                list_others.append(image)

        return list_personal, list_others

    def compare_image_list(self, group):
        list_personal, list_others = self.assign_groups_for_comparision(group)
        for image_personal in list_personal:
            for image_others in list_others:
                self.compare_image_descriptors(image_personal, image_others)

    def compare_image_descriptors(self, image_data_personal, image_data_others):
        distance = dist.euclidean(
            image_data_personal._descriptor, image_data_others._descriptor
        )
        #
        if float(distance) <= float(0.9) or distance == 0.0:
            self._matches.append((image_data_personal, image_data_others))


class ResultPrinter:
    def __init__(self, window_title, image_object_list, rows, cols) -> None:
        self._window_title = window_title
        self._images = image_object_list
        self._rows = rows
        self._cols = cols
        self._splited_images = []
        if len(self._images) > 24:
            self.split_image_list()

    def split_image_list(self):
        counter = 0
        temp = []
        for image_object in self._images:
            if counter == 10:
                self._splited_images.append(temp)
                counter = 0
                temp = []
                temp.append(image_object)
                counter += 1
            else:
                temp.append(image_object)
                counter += 1
        self._splited_images.append(temp)
        print(len(self._splited_images[0]))

    def run(self):
        if self._splited_images:
            counter = 1
            for image_list in self._splited_images:
                figure = plt.figure(self._window_title + f" {counter}")
                for i in range(len(image_list)):
                    figure.add_subplot(self._rows, self._cols, i + 1)
                    plt.imshow(image_list[i].scaled)
                    ax = plt.gca()
                    ax.get_yaxis().set_visible(False)
                    ax.get_xaxis().set_visible(False)
                counter += 1

                plt.show(block=True)
        else:
            counter = 1
            for image_object in self._images:
                figure = plt.figure(self._window_title)
                figure.add_subplot(self._rows, self._cols, counter)
                plt.imshow(image_object.scaled)
                ax = plt.gca()
                ax.get_yaxis().set_visible(False)
                ax.get_xaxis().set_visible(False)
                counter += 1

            plt.show(block=True)



personal_images = os.listdir("personal")

others_images = os.listdir("others")

personal = DataSet(personal_images, "personal")

others = DataSet(others_images, "others")


gs = GroupAssigner(personal, others, 5)
gs.run()

image_list = []
for image in personal.images:
    image_list.append(image)
rp = ResultPrinter("Zbiór oryginałów", image_list, 5,3)
rp.run()

image_list = []
for image in others.images:
    image_list.append(image)
rp = ResultPrinter("Zbiór duplikatów", image_list, 5,3)
rp.run()



rp = ResultPrinter("Ciemne obrazy", gs.g1, 5, 3)
rp.run()

rp = ResultPrinter("Umiarkowanie jasne obrazy", gs.g3, 5, 3)
rp.run()

rp = ResultPrinter("Jasne obrazy", gs.g2, 5, 3)
rp.run()


gd = ZernikeMomentDescriptorGenerator(personal, others)
gd.run()
start = time.time()
im = ZenrickeMomentImageMatcher(gs.g1, gs.g2, gs.g3)
matches = im.run()
print(f"Matches found: {len(matches)}")
end = time.time()
match_time = np.longdouble(end - start)
print(f"Execution time: {match_time:e}")
image_list = []
for i in range(len(matches)):
    image_list.append(matches[i][0])
    image_list.append(matches[i][1])

rp = ResultPrinter("Dopasowania", image_list,12,2)
rp.run()