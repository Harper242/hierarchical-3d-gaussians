import os
import argparse
import numpy as np
from exif import Image
from sklearn.neighbors import NearestNeighbors


def decimal_coords(coords, ref):
    decimal_degrees = coords[0] + coords[1] / 60 + coords[2] / 3600
    if ref == "S" or ref == "W":
        decimal_degrees = -decimal_degrees
    return decimal_degrees


def image_coordinates(image_name):
    # image_name 是相对於 args.image_path 的路徑
    image_path = os.path.join(args.image_path, image_name)
    if not os.path.exists(image_path):
        return None

    with open(image_path, "rb") as src:
        img = Image(src)
    if img.has_exif:
        try:
            img.gps_longitude
            coords = [
                decimal_coords(img.gps_latitude, img.gps_latitude_ref),
                decimal_coords(img.gps_longitude, img.gps_longitude_ref),
            ]
            return coords
        except AttributeError:
            return None
    else:
        return None


def get_matches(img_name, cam_center, cam_nbrs, img_names_gps):
    if cam_nbrs is None or len(img_names_gps) == 0:
        return []

    _, indices = cam_nbrs.kneighbors(cam_center[None])
    lines = []
    # indices[0, 0] 是自己 本身 避免自匹配 從 1 開始
    for idx in indices[0, 1:]:
        lines.append(f"{img_name} {img_names_gps[idx]}\n")
    return lines


def find_images_names(root_dir):
    image_files_by_subdir = []

    for dirpath, dirnames, filenames in os.walk(root_dir):
        image_files = sorted(
            [f for f in filenames if f.lower().endswith((".png", ".jpg", ".jpeg", ".JPG", ".PNG"))]
        )

        if image_files:
            rel_dir = os.path.relpath(dirpath, root_dir)
            if rel_dir == ".":
                rel_dir = ""
            image_files_by_subdir.append(
                {
                    "dir": rel_dir,   # 相對於 root_dir 的子目錄 可以為空字串
                    "images": image_files,
                }
            )

    return image_files_by_subdir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--n_seq_matches_per_view", default=0, type=int)
    parser.add_argument("--n_quad_matches_per_view", default=10, type=int)
    parser.add_argument("--n_loop_closure_match_per_view", default=5, type=int)
    parser.add_argument("--loop_matches", nargs="*", default=[], type=int)
    parser.add_argument("--n_gps_neighbours", default=25, type=int)
    args = parser.parse_args()

    # 迴圈閉合對 形如 [i0 j0 i1 j1 ...] 轉為 Nx2
    loop_matches = np.array(args.loop_matches, dtype=np.int64).reshape(-1, 2)

    loop_rel_matches = np.arange(0, args.n_loop_closure_match_per_view)
    loop_rel_matches = 2 ** loop_rel_matches
    loop_rel_matches = np.concatenate(
        [-loop_rel_matches[::-1], np.array([0]), loop_rel_matches]
    )

    image_files_organised = find_images_names(args.image_path)

    matches_str = []

    def add_match(current_cam, matched_cam, current_image_file, matched_frame_id):
        if matched_frame_id < len(matched_cam["images"]):
            matched_image_file = matched_cam["images"][matched_frame_id]

            cur_dir = current_cam["dir"]
            match_dir = matched_cam["dir"]

            if cur_dir:
                img1 = f"{cur_dir}/{current_image_file}"
            else:
                img1 = current_image_file

            if match_dir:
                img2 = f"{match_dir}/{matched_image_file}"
            else:
                img2 = matched_image_file

            matches_str.append(f"{img1} {img2}\n")

    # 時序匹配 和 指數步長匹配
    for cam_idx, current_cam in enumerate(image_files_organised):
        for matched_cam_idx, matched_cam in enumerate(image_files_organised[cam_idx:]):
            for current_image_id, current_image_file in enumerate(current_cam["images"]):
                for frame_step in range(args.n_seq_matches_per_view):
                    matched_frame_id = current_image_id + frame_step
                    add_match(current_cam, matched_cam, current_image_file, matched_frame_id)

                for match_id in range(args.n_quad_matches_per_view):
                    frame_step = args.n_seq_matches_per_view + int(2 ** match_id) - 1
                    matched_frame_id = current_image_id + frame_step
                    add_match(current_cam, matched_cam, current_image_file, matched_frame_id)

            # 迴圈閉合匹配
            for loop_match in loop_matches:
                for current_loop_rel_match in loop_rel_matches:
                    current_image_id = loop_match[0] + current_loop_rel_match
                    if 0 <= current_image_id < len(current_cam["images"]):
                        current_image_file = current_cam["images"][current_image_id]
                        for matched_loop_rel_match in loop_rel_matches:
                            matched_frame_id = loop_match[1] + matched_loop_rel_match
                            add_match(current_cam, matched_cam, current_image_file, matched_frame_id)

    # GPS 匹配
    if args.n_gps_neighbours > 0:
        all_img_names = []
        for cam in image_files_organised:
            if cam["dir"]:
                all_img_names += [os.path.join(cam["dir"], img_name) for img_name in cam["images"]]
            else:
                all_img_names += [img_name for img_name in cam["images"]]

        all_cam_centers = [image_coordinates(img_name) for img_name in all_img_names]

        img_names_gps = [
            img_name
            for img_name, cam_center in zip(all_img_names, all_cam_centers)
            if cam_center is not None
        ]
        cam_centers_gps = [
            cam_center for cam_center in all_cam_centers if cam_center is not None
        ]

        if len(cam_centers_gps) > 0:
            cam_centers = np.array(cam_centers_gps)
            cam_nbrs = NearestNeighbors(
                n_neighbors=min(args.n_gps_neighbours, len(cam_centers_gps))
            ).fit(cam_centers)
        else:
            cam_nbrs = None
            cam_centers = np.empty((0, 2))

        for img_name, cam_center in zip(img_names_gps, cam_centers):
            matches_str.extend(
                get_matches(img_name, cam_center, cam_nbrs, img_names_gps)
            )

    # 去重 和 移除互為對稱的匹配對
    intermediate_out_matches = list(dict.fromkeys(matches_str))

    reciproc_matches = []
    for match in intermediate_out_matches:
        a, b = match.strip().split()
        reciproc_matches.append(f"{b} {a}\n")

    reciproc_matches_dict = dict.fromkeys(reciproc_matches)

    out_matches = [
        match for match in intermediate_out_matches if match not in reciproc_matches_dict
    ]

    with open(args.output_path, "w") as f:
        f.write("".join(out_matches))

    print(0)
