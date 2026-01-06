import skimage

from src.utils.config_loader import load_settings
import numpy as np
from matplotlib import pyplot as plt
import cv2


class StraighteningService:
    def __init__(self):
        self.settings = load_settings()
        self.straightening_settings = self.settings["straightening_settings"]

    def straighten_image(self, img:np.ndarray, masks):

        r_angle, eps_corners, n_points, circle_diameter, max_angle_theta, points_short_amount, show_contour_plot \
            = self.straightening_settings.values()

        c_img, c_mask = _rotate_and_crop(img, masks[0])
        c_img, c_mask = _rotate_and_crop(c_img, c_mask, r_angle)


        points_tight = _interpolate_contour(c_mask, n_points)
        corners_tight = _find_corners(points_tight, eps_corners)
        points_processed, corners_processed = _chamfer_det_fix(points_tight, corners_tight, circle_diameter,
                                                               max_angle_theta, eps_corners)
        points_processed = _roll_contour(points_processed, corners_processed)
        img_preprocessed = _remove_background(c_img, points_tight)
        if len(corners_processed) > 4:
            angles_temp = np.abs(_find_corner_angles(points_processed, corners_processed, circle_diameter) - 90)
            angles_indices = np.argsort(angles_temp, axis=0)[4:]
            corners_processed = np.delete(corners_processed, angles_indices.reshape(-1), axis=0)
            print('Warning: more than 4 corners detected')
        corners_on_shape, corners_indices = _find_nearest_indices(points_processed, corners_processed)
        straights = _get_straights(points_processed, corners_indices)
        points_warped, points_destination = _generate_warped_mesh(img_preprocessed, straights, points_short_amount)

        tform = skimage.transform.PiecewiseAffineTransform()
        tform.estimate(points_destination, points_warped)
        warped = skimage.transform.warp(img_preprocessed, tform)

        if show_contour_plot:
            fig, ax = plt.subplots(figsize=(15, 15))
            ax.imshow(img_preprocessed)
            plt.plot(points_processed[corners_indices, 0], points_processed[corners_indices, 1], 'bo')
            plt.plot()
            for i, corner in enumerate(corners_processed):
                cnt_cv = points_processed.reshape((-1, 1, 2)).astype(np.int32)
                r_circle = cv2.arcLength(cnt_cv, True) * circle_diameter
                circle = plt.Circle((corner), r_circle, color='g', fill=False)
                txt = f'{i}'
                ax.text(corner[0], corner[1], txt, color='k', fontweight='bold', fontsize=12)
                ax.add_patch(circle)

            U, V = (points_destination - points_warped).T
            plt.quiver(points_warped[:, 0], points_warped[:, 1], U, V, angles='xy', scale_units='xy', scale=.5)

            ax.axis('off')
            plt.show()

            plt.figure(figsize=(20, 20))
            plt.imshow(warped)
            plt.show()

        return warped

    def straighten_simple(self, img:np.ndarray, masks:np.ndarray, verbose:bool=False):
        r_angle, eps_corners, n_points, circle_diameter, max_angle_theta, points_short_amount, show_contour_plot \
            = self.straightening_settings.values()
        c_img, c_mask = _rotate_and_crop(img, masks[0])
        points_tight = _interpolate_contour(c_mask, n_points)
        img_preprocessed = _remove_background(c_img, points_tight)
        if verbose:
            fig, ax = plt.subplots(figsize=(15, 15))
            ax.imshow(img_preprocessed)
        return img_preprocessed




def _rotate_and_crop(img, mask, angle_in = None):

    rect = cv2.minAreaRect(mask.astype(np.float32))
    center, size, angle = rect
    angle = angle + 90.0
    (w, h) = size
    if angle_in is not None:
        angle = angle_in

    (H, W) = img.shape[:2]
    M = cv2.getRotationMatrix2D(center=center, angle=angle, scale=1.0)

    cos, sin = np.abs(M[0,0]), np.abs(M[0,1])
    new_W = int(H*sin + W*cos)
    new_H = int(H*cos + W*sin)

    M[0,2] += (new_W/2 - center[0])
    M[1,2] += (new_H/2 - center[1])

    rotated_img = cv2.warpAffine(img, M, (new_W, new_H))

    ones = np.ones((mask.shape[0], 1))
    mask_h = np.hstack([mask, ones])  # Nx3
    rotated_mask = (M @ mask_h.T).T   # Nx2

    x_min, y_min = rotated_mask.min(axis=0).astype(int)
    x_max, y_max = rotated_mask.max(axis=0).astype(int)

    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(new_W, x_max)
    y_max = min(new_H, y_max)

    cropped_img = rotated_img[y_min:y_max, x_min:x_max]

    cropped_mask = rotated_mask - np.array([x_min, y_min])
    return cropped_img, cropped_mask

def _rotate_image(img, angle):

    image = img.copy()

    if angle == 90:
        rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180:
        rotated = cv2.rotate(image, cv2.ROTATE_180)
    elif angle == 270:
        rotated = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif angle == 0:
        rotated = image
    else:
        raise ValueError("Angle must be 90, 180, or 270")
    return rotated

def _rotate_image_and_bbox(image, bbox, angle_deg):
    h, w = image.shape[:2]
    center = (w // 2, h // 2)

    # rotation matrix
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)

    # rotate image
    rotated_image = _rotate_image(image, angle_deg)

    # rotate bbox (homogeneous coords)
    bbox_h = np.hstack([bbox, np.ones((bbox.shape[0], 1))])
    rotated_bbox = bbox_h @ M.T

    return rotated_image, rotated_bbox

def _interpolate_contour(contour, n_points = 300):
    dtype = contour.dtype

    new_points = np.empty((0, 2), dtype=dtype)
    c_mask_cv = contour.reshape((-1,1,2)).astype(np.int32)

    len_shape = cv2.arcLength(c_mask_cv, True)
    min_dist = len_shape // n_points

    for i in range(0, len(contour)):
        p1 = contour[i - 1, :]
        p2 = contour[i, :]

        dist = np.linalg.norm(p2 - p1)
        if dist > min_dist:
            n_pts = (dist // min_dist).astype(np.int32)
            for n in range(n_pts):
                frac = n/n_pts
                new_pt = p1 * (1 - frac) + p2 * frac
                new_points = np.append(new_points, [new_pt], axis=0)
        new_points = np.append(new_points, [p2], axis=0)

    return new_points

def _find_corners(contour, eps=.01):
    pts_cv = contour.reshape((-1, 1, 2)).astype(np.int32)
    epsilon = eps * cv2.arcLength(pts_cv, True)
    approx = cv2.approxPolyDP(pts_cv, epsilon, True)
    corners = approx.reshape((-1, 2))

    return corners

def _unchamfer_point(corners, angles, indices):
    c0 = corners[indices[0]]
    c1 = corners[indices[1]]

    t0 = angles[indices[0]]
    t1 = angles[indices[1]]

    d = np.linalg.norm(c1 - c0)

    alpha = np.radians(180 - t0)
    beta = np.radians(180 - t1)
    gamma = np.pi - alpha - beta

    m = d * (np.sin(beta) / np.sin(gamma))
    l = d * (np.sin(alpha) / np.sin(gamma))

    u = (c1 - c0)/d
    n = np.array([-u[1], u[0]])

    a = (m**2 - l**2 + d**2) / (2*d)
    h = (m ** 2 - a ** 2) ** .5

    P = c0 + (a * u)
    C_out1 = P + (h*n)
    C_out2 = P - (h*n)

    cross_z = lambda a, b, c: (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])
    cross_1 = cross_z(c0, c1, C_out1)
    cross_2 = cross_z(c0, c1, C_out2)


    if cross_1 < 0:
        return C_out2
    else:
        return C_out1

def _contour_circle_intersections(contour, center, radius):
    contour = np.squeeze(contour)
    cx, cy = center
    N = len(contour)
    intersections = []

    for i in range(N):
        x1, y1 = contour[i]
        x2, y2 = contour[(i + 1) % N]  # next point (closed loop)
        dx, dy = x2 - x1, y2 - y1

        # Quadratic coefficients for line-circle intersection
        a = dx**2 + dy**2
        b = 2 * (dx * (x1 - cx) + dy * (y1 - cy))
        c = (x1 - cx)**2 + (y1 - cy)**2 - radius**2

        disc = b**2 - 4 * a * c
        if disc < 0:
            continue  # no intersection

        sqrt_disc = np.sqrt(disc)
        for sign in [-1, 1]:
            t = (-b + sign * sqrt_disc) / (2 * a)
            if 0 <= t <= 1:  # lies on the segment
                xi = x1 + t * dx
                yi = y1 + t * dy
                intersections.append((xi, yi))


    if len(np.array(intersections)) > 2:
        #print(f'Multiple intersections found for {center}')
        #print(np.array(intersections))
        pts = np.asarray(intersections)
        tol = 5
        grid = np.round(pts / tol).astype(int)
        #print(grid)
        _, idx = np.unique(grid, return_index=True, axis=0)
        #print(idx)
        intersections = pts[np.sort(idx)]
        #print(f'fixed intersections: {intersections}')

    if len(np.array(intersections)) == 1:
        pts = np.asarray(intersections)
        intersections = np.vstack([pts, pts])

    return intersections



def _find_corner_angles(contour, corners, r_circle=.015, dbg=False):
    angles = np.empty((len(corners), 1))
    cnt_cv = contour.reshape((-1,1,2)).astype(np.int32)
    r_circle = cv2.arcLength(cnt_cv, True) * r_circle
    for n, vertex in enumerate(corners):
        if dbg: print(vertex)
        intersections = _contour_circle_intersections(contour, vertex, r_circle)
        intersections = np.array(intersections)
        if dbg:
            print(intersections)
        v1 = vertex - intersections[0, :]
        v2 = vertex - intersections[1, :]
        top = np.dot(v1, v2)
        bot = np.linalg.norm(v1) * np.linalg.norm(v2)
        theta = np.degrees(np.arccos(top/bot))
        angles[n] = theta

    return angles

def _linspace2d(a, b, k, include_end=True):
        a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
        if k <= 1:
            return np.vstack([a, b]) if include_end else np.array([a])
        t = np.linspace(0.0, 1.0, k, endpoint=include_end)
        return (1.0 - t)[:, None] * a + t[:, None] * b

def _interpolate_chamfer(contour, corners, angles, indices, r_circle, eps_corners):

    t1 = corners[indices[0]]
    t2 = corners[indices[1]]


    c_mask_cv = contour.reshape((-1,1,2)).astype(np.int32)
    len_shape = cv2.arcLength(c_mask_cv, True)
    resolution_chamfer = len_shape // len(contour)

    i1 = _find_point_index(contour, t1)
    i2 = _find_point_index(contour, t2)

    C_out = _unchamfer_point(corners, angles, indices)

    min_i = min(i1, i2)
    max_i = max(i1, i2)

    corner1 = contour[min_i, :]
    corner2 = contour[max_i, :]

    n_pts_1 = (np.linalg.norm(C_out - corner1) // int(resolution_chamfer)).astype(int)
    n_pts_2 = (np.linalg.norm(C_out - corner2) // int(resolution_chamfer)).astype(int)

    seg1 = _linspace2d(corner1, C_out, n_pts_1, include_end=True)
    seg2 = _linspace2d(C_out, corner2, n_pts_2, include_end=True)

    begin_shape = contour[:min_i]
    end_shape = contour[max_i+1:]


    shape_messy = np.vstack([begin_shape, seg1, seg2[1:], end_shape])
    h, w = np.max(shape_messy[:,1]).astype(int)+5, np.max(shape_messy[:,0]).astype(int)+5
    mask = np.zeros((h, w), dtype=np.uint8)

    cv2.fillPoly(mask, [shape_messy.astype(np.int32)], 255)
    shapes, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    shape_clean = max(shapes, key=cv2.contourArea).reshape(-1, 2)

    corners = _find_corners(shape_clean, eps_corners)
    angles = _find_corner_angles(shape_clean, corners, r_circle)

    return shape_clean, corners, angles

def _find_point_index(points, target, tol=1):
    mask = np.all(np.isclose(points, target, atol=tol), axis=1)
    return np.where(mask)[0][-1]

def _fix_chamfer_wrapped(contour, corners, circle_diameter, max_angle_theta, eps_corners):
    angles = _find_corner_angles(contour, corners, circle_diameter)
    for n in range(len(corners)):
        pair = np.array((angles[n-1], angles[n]))
        angle_test = np.abs(pair - 90)
        check_out_dimension = np.all(angle_test > max_angle_theta)
        if check_out_dimension:
            contour, corners, _ = _interpolate_chamfer(contour, corners, angles, [n - 1, n], circle_diameter, eps_corners)
            return contour, corners, False

    return contour, corners, True

def _fix_chamfer(contour, corners, circle_diameter, max_angle_theta, eps_corners):
    for n in range(len(corners)):
        contour, corners, finished_flag = _fix_chamfer_wrapped(contour, corners, circle_diameter, max_angle_theta, eps_corners)
        if finished_flag:
            return contour, corners

    return contour, corners

def _remove_background(img, contour):
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    white_bg = np.ones_like(img, dtype=np.uint8) * 255
    cv2.fillPoly(mask, [contour.astype(np.int32)], 255)
    masked = cv2.bitwise_and(img, img, mask=mask)
    bg = cv2.bitwise_and(white_bg, white_bg, mask=~mask)
    img = cv2.add(masked, bg)
    return img

def _find_nearest_indices(contours, corners):
    nearest_index = np.zeros_like(corners)
    nearest_indices = np.zeros(len(corners)).astype(np.int32)
    for row, corner in enumerate(corners):
        dists = np.sum((contours - corner)**2, axis=1)
        min_dist = np.argmin(dists)
        nearest_index[row] = contours[min_dist]
        nearest_indices[row] = min_dist

    return nearest_index, nearest_indices

def _get_straights(contours, corner_indices):
    straights = []
    for i in range(len(corner_indices)):
        i1 = corner_indices[i - 1]
        i2 = corner_indices[i]
        if i1 > i2:
            straight = np.vstack([contours[i1:], contours[:i2]])
            #straight = np.r_[contours[:i2], contours[i1:]]
        else:
            straight = contours[i1:i2]
        straights.append(straight)
    return straights

def _roll_contour(contour, corners):
    r_flag = False
    if np.all(np.isclose(contour[0, :], contour[-1, :])):
        contour = contour[:-1]

    c0 = corners[0].reshape(1, 2)
    f_idx = _find_point_index(contour, c0)
    if r_flag: contour = contour[:-1]
    contour = np.roll(contour, -f_idx, axis=0)
    if r_flag: contour = np.vstack([contour, c0])
    return contour

def _chamfer_det_fix(contour, corners, circle_diameter, max_angle_theta=20, eps_corners=.01):
    if len(corners) > 4:
        contour, corners = _fix_chamfer(contour, corners, circle_diameter, max_angle_theta, eps_corners)
    return contour, corners

def _line_polyline_intersections(polyline: np.ndarray, value: float, orientation: str = "horizontal"):
    """
    Find the nearest intersection between a straight line and a polyline.
    If no intersection is found, returns the closest point on the polyline.

    Args:
        polyline (np.ndarray): Array of shape (n, 2), sequence of (x, y) points.
        value (float): The constant value for the line.
            - If orientation == "horizontal", line is y = value.
            - If orientation == "vertical", line is x = value.
        orientation (str): "horizontal" or "vertical".

    Returns:
        (x, y) tuple for the nearest intersection or closest point.
    """
    intersections = []
    for (x1, y1), (x2, y2) in zip(polyline[:-1], polyline[1:]):
        if orientation == "horizontal":
            if (y1 - value) * (y2 - value) <= 0 and y1 != y2:
                t = (value - y1) / (y2 - y1)
                x = x1 + t * (x2 - x1)
                intersections.append((float(x), float(value)))
        elif orientation == "vertical":
            if (x1 - value) * (x2 - value) <= 0 and x1 != x2:
                t = (value - x1) / (x2 - x1)
                y = y1 + t * (y2 - y1)
                intersections.append((float(value), float(y)))
        else:
            raise ValueError("orientation must be 'horizontal' or 'vertical'")

    if intersections:
        # choose nearest to line
        if orientation == "horizontal":
            # distance is just |x - polyline centroid x| (or min |x|)
            idx = np.argmin([abs(x) for x, _ in intersections])
        else:  # vertical
            idx = np.argmin([abs(y) for _, y in intersections])
        return intersections[idx]

    # fallback: nearest polyline point to the line
    if orientation == "horizontal":
        idx = np.argmin(np.abs(polyline[:, 1] - value))
    else:  # vertical
        idx = np.argmin(np.abs(polyline[:, 0] - value))
    return tuple(map(float, polyline[idx]))

def _generate_warped_mesh(img, straights, short_axis_point_amount):
    points_short_amount = short_axis_point_amount

    l_straights = [len(i) for i in straights]
    diff_straights = np.array([np.abs(np.sum(np.diff(i, axis=0), axis=0)) for i in straights])
    sum_straights = np.array([np.sum(i, axis = 0) for i in straights])
    len_straights = np.array([np.linalg.norm(i, axis=0) for i in diff_straights])
    # left - small x sum, from big y diff sum
    # right - big x sum, from big y diff sum
    # bot - big y sum, from small y diff sum
    # top - small y sum, from small y diff sum
    # compute straight diffs and sums
    diffs = np.array([np.abs(np.sum(np.diff(s, axis=0), axis=0)) for s in straights])
    sums  = np.array([np.sum(s, axis=0) for s in straights])
    # split by x-diff (horizontal vs vertical candidates)
    idx_sorted = np.argsort(diffs[:, 0])
    idx_ver, idx_hor = np.split(idx_sorted, 2)
    # pick left/right from vertical indices
    left, right = [straights[i] for i in idx_ver[np.argsort(sums[idx_ver, 0])]]
    # now pick top/bot
    top, bot = [straights[i] for i in idx_hor[np.argsort(sums[idx_hor, 1])]]
    # find mean lenghts
    mean_lens = np.array([
        len_straights[idx_ver].mean(),
        len_straights[idx_hor].mean()
    ])


    h_img, w_img = img.shape[:2]
    widths = np.linspace(0, w_img, points_short_amount).astype(int)
    num_heights = h_img // (widths[1] - widths[0]).astype(int)
    heights = np.linspace(0, h_img, num_heights).astype(int)

    if mean_lens[0] > mean_lens[1]: # vertical orientation
        print('vert')
        widths = np.linspace(0, w_img, points_short_amount).astype(int)
        num_heights = h_img // (widths[1] - widths[0]).astype(int)
        heights = np.linspace(0, h_img, num_heights).astype(int)
    else:
        print('hor')
        heights = np.linspace(0, h_img, points_short_amount).astype(int)
        num_widths = w_img // (heights[1] - heights[0]).astype(int)
        widths = np.linspace(0, w_img, num_widths).astype(int)


    points_destination = np.empty((len(heights) * len(widths), 2))
    points_warped = np.empty((len(heights) * len(widths), 2))
    n = 0
    for i, hei in enumerate(heights):
        for j, wid in enumerate(widths):
            points_destination[n, :] = [wid, hei]
            intersect_left = _line_polyline_intersections(left, hei, orientation="horizontal")[0]
            intersect_right = _line_polyline_intersections(right, hei, orientation="horizontal")[0]
            intersect_top = _line_polyline_intersections(top, wid, orientation="vertical")[1]
            intersect_bot = _line_polyline_intersections(bot, wid, orientation="vertical")[1]

            x_step = wid/widths[-1]
            y_step = hei/heights[-1]

            x_value = (((1 - x_step) * intersect_left) + (x_step * intersect_right)).astype(int)
            y_value = (((1 - y_step) * intersect_top) + (y_step * intersect_bot)).astype(int)
            points_warped[n, :] = [x_value, y_value]
            n+=1
    return points_warped, points_destination


if __name__ == "__main__":
    from segmenting_service import Segmentator
    yolo = Segmentator()
    service = StraighteningService()

    dupa = yolo.segment('/home/maciejka/Documents/projects/grosia_app/02_test_images/1.jpg')
    essa = service.straighten_image('/home/maciejka/Documents/projects/grosia_app/02_test_images/1.jpg', dupa)
    plt.imshow(essa)
    plt.show()