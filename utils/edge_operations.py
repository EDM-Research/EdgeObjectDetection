import cv2
import numpy as np
import random
import os
import utils.edge_config as config


def random_canny(image: np.array) -> np.array:
    lower = random.randint(config.canny_base - config.canny_margin, config.canny_base + config.canny_margin)
    upper = random.randint(lower + config.canny_space - config.canny_margin, lower + config.canny_space + config.canny_margin)
    return cv2.Canny(image, lower, upper)


def drop_edges(edges: np.array) -> np.array:
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    pixel_count = cv2.countNonZero(edges)
    drop_percentage = (random.uniform(0.0, 0.4) + 0.2) * config.drop_factor
    gaps = random.randint(3, 6)
    mean_gap_size = (pixel_count * drop_percentage)/gaps
    gap_start_indices = random.sample(range(0, pixel_count), gaps)

    gap_indices = []
    for gap_start in gap_start_indices:
        gap_size = int(random.gauss(mean_gap_size, mean_gap_size/2))
        gap_indices.extend(list(range(gap_start, gap_start + gap_size)))

    index = 0
    result = edges.copy()
    for contour in contours:
        for i in range(len(contour)):
            point = contour[i][0]
            if index in gap_indices:
                result[point[1], point[0]] = 0
            index += 1

    return result


def _drop_edges(edges: np.array) -> np.array:
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    result = np.zeros(edges.shape, dtype=edges.dtype)

    drop_edge_prob = random.uniform(0.0, 0.3) + 0.1

    for contour in contours:
        if random.uniform(0.0, 1.0) < drop_edge_prob:
            continue

        for i in range(len(contour)):
            point = contour[i][0]
            result[point[1], point[0]] = 255

    return result


def deform_edges(edges: np.array, deform_factor: 0.5) -> np.array:
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    used_check = np.zeros(edges.shape, dtype=bool)
    result = np.zeros(edges.shape, dtype=edges.dtype)

    for contour in contours:

        remove_prob = (random.uniform(0.0, 0.30) + 0.01) * deform_factor
        skip_prob = (random.uniform(0.0, 0.40) + 0.75) * deform_factor
        offset_prob = (random.uniform(0.0, 0.80) + 0.45) * deform_factor
        offset_amount = random.randint(-2, 2)

        last_one_added = True

        for i in range(len(contour)):
            point = contour[i][0]
            if random.uniform(0.0, 1.0) < offset_prob:
                offset_amount = random.randint(-1, 1)

            if last_one_added:
                skip = random.uniform(0.0, 1.0) < remove_prob
            else:
                skip = random.uniform(0.0, 1.0) < skip_prob

            if not skip:
                if not used_check[point[1], point[0]]:
                    used_check[point[1], point[0]] = True

                    offset_dir = contour[i + 1][0] - point if i < len(contour) - 1 else [0, 0]

                    pos = [point[0]+offset_dir[0]*offset_amount, point[1]+offset_dir[1]*offset_amount]
                    if 0 <= pos[1] < result.shape[0] and 0 <= pos[0] < result.shape[1]:
                        result[pos[1], pos[0]] = 255
                last_one_added = True
            else:
                last_one_added = False

    return result


if __name__ == "__main__":
    real_image_data = "D:/Datasets/DIMO/dimo/real_jaigo_000-150"
    for folder in os.listdir(real_image_data):
        if os.path.isdir(os.path.join(real_image_data, folder)):
            scene = int(folder)
            sim_scene_id = f"{str(scene).zfill(6)}_00"
            real_image_folder = os.path.join(real_image_data, folder, "rgb")
            sim_image_folder = os.path.join("D:/Datasets/DIMO/dimo/sim_jaigo_real_light_real_pose", sim_scene_id, "rgb")

            for real_image_file, sim_image_file in zip(os.listdir(real_image_folder), os.listdir(sim_image_folder)):
                real_image = cv2.imread(os.path.join(real_image_folder, real_image_file))
                real_image = cv2.resize(real_image, (real_image.shape[1] // 2, real_image.shape[0] // 2))

                real_edges = random_canny(real_image)
                #real_edges = cv2.Canny(real_image, config.canny_base, config.canny_base + config.canny_space)

                sim_image = cv2.imread(os.path.join(sim_image_folder, sim_image_file))
                sim_image = cv2.resize(sim_image, (sim_image.shape[1] // 2, sim_image.shape[0] // 2))

                sim_edges = random_canny(sim_image)
                #sim_edges = cv2.Canny(sim_image, config.canny_base, config.canny_base + config.canny_space)

                deformed = deform_edges(sim_edges, 0.5)

                result = np.hstack((real_edges, sim_edges, deformed))
                result = cv2.resize(result, (result.shape[1] // 2, result.shape[0] // 2))
                cv2.imshow("result", result)
                cv2.waitKey(0)

