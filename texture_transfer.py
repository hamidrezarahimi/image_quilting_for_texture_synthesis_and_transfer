import numpy as np
from skimage import util
import math
from skimage.color import rgb2gray
from skimage.filters import gaussian
import heapq
from tqdm import tqdm
from skimage import io


class TextureTransfer:
    def __init__(self, texture, target, patch_length):
        self.texture = util.img_as_float(texture)[:, :, :3]

        target_float = util.img_as_float(target)
        if len(target_float.shape) == 2 or target_float.shape[2] == 1:
            target_float = color.gray2rgb(target_float)
        self.target = target_float[:, :, :3]

        self.corr_target = rgb2gray(target_float)
        self.h, self.w, _ = self.target.shape
        self.patch_length = patch_length
        self.corr_texture = rgb2gray(texture)
        self.overlap = patch_length // 6

    def best_corr_patch(self, y, x):
        h, w, _ = self.texture.shape
        errors = np.zeros((h - self.patch_length, w - self.patch_length))

        corr_target_patch = self.corr_target[y:y + self.patch_length, x:x + self.patch_length]
        cur_patch_height, cur_patch_width = corr_target_patch.shape

        for i in range(h - self.patch_length):
            for j in range(w - self.patch_length):
                corr_texture_patch = self.corr_texture[i:i + cur_patch_height, j:j + cur_patch_width]
                e = corr_texture_patch - corr_target_patch
                errors[i, j] = np.sum(e ** 2)

        i, j = np.unravel_index(np.argmin(errors), errors.shape)
        return self.texture[i:i + cur_patch_height, j:j + cur_patch_width]

    def best_corr_overlap_patch(self, res, y, x, alpha=0.1, level=0):
        h, w, _ = self.texture.shape
        errors = np.zeros((h - self.patch_length, w - self.patch_length))

        corr_target_patch = self.corr_target[y:y + self.patch_length, x:x + self.patch_length]
        di, dj = corr_target_patch.shape

        for i in range(h - self.patch_length):
            for j in range(w - self.patch_length):
                patch = self.texture[i:i + di, j:j + dj]
                l2error = self.l2_overlap_diff(patch, res, y, x)
                overlap_error = np.sum(l2error)

                corr_texture_patch = self.corr_texture[i:i + di, j:j + dj]
                corr_error = np.sum((corr_texture_patch - corr_target_patch) ** 2)

                prev_error = 0
                if level > 0:
                    prev_error = patch[self.overlap:, self.overlap:] - res[y + self.overlap:y + self.patch_length,
                                                                       x + self.overlap:x + self.patch_length]
                    prev_error = np.sum(prev_error ** 2)

                errors[i, j] = alpha * (overlap_error + prev_error) + (1 - alpha) * corr_error

        i, j = np.unravel_index(np.argmin(errors), errors.shape)
        return self.texture[i:i + di, j:j + dj]

    def l2_overlap_diff(self, patch, res, y, x):
        dy, dx, _ = patch.shape
        error = 0

        if x > 0:
            left = patch[:, :self.overlap] - res[y:y + dy, x:x + self.overlap]
            error += np.sum(left ** 2)

        if y > 0:
            up = patch[:self.overlap, :] - res[y:y + self.overlap, x:x + dx]
            error += np.sum(up ** 2)

        if x > 0 and y > 0:
            corner = patch[:self.overlap, :self.overlap] - res[y:y + self.overlap, x:x + self.overlap]
            error -= np.sum(corner ** 2)

        return error

    def min_cut_patch(self, patch, res, y, x):
        patch = patch.copy()
        dy, dx, _ = patch.shape
        min_cut = np.zeros_like(patch, dtype=bool)

        if x > 0:
            left = patch[:, :self.overlap] - res[y:y + dy, x:x + self.overlap]
            left_l2 = np.sum(left ** 2, axis=2)
            for i, j in enumerate(self.min_cut_path(left_l2)):
                min_cut[i, :j] = True

        if y > 0:
            up = patch[:self.overlap, :] - res[y:y + self.overlap, x:x + dx]
            up_l2 = np.sum(up ** 2, axis=2)
            for j, i in enumerate(self.min_cut_path(up_l2.T)):
                min_cut[:i, j] = True

        np.copyto(patch, res[y:y + dy, x:x + dx], where=min_cut)
        return patch

    def min_cut_path(self, errors):
        # Dijkstra's algorithm vertical
        pq = [(error, [i]) for i, error in enumerate(errors[0])]
        heapq.heapify(pq)

        h, w = errors.shape
        seen = set()

        while pq:
            error, path = heapq.heappop(pq)
            cur_depth = len(path)
            cur_index = path[-1]

            if cur_depth == h:
                return path

            for delta in -1, 0, 1:
                next_index = cur_index + delta

                if 0 <= next_index < w:
                    if (cur_depth, next_index) not in seen:
                        cum_error = error + errors[cur_depth, next_index]
                        heapq.heappush(pq, (cum_error, path + [next_index]))
                        seen.add((cur_depth, next_index))

    def transfer(self, mode="cut", alpha=0.1, level=0, prior=None, blur=False):
        if blur:
            self.corr_texture = gaussian(self.corr_texture, sigma=3)
            self.corr_target = gaussian(self.corr_target, sigma=3)

        if level == 0:
            res = np.zeros_like(self.target)
        else:
            res = prior

        num_patches_high = math.ceil((self.h - self.patch_length) / (self.patch_length - self.overlap)) + 1 or 1
        num_patches_wide = math.ceil((self.w - self.patch_length) / (self.patch_length - self.overlap)) + 1 or 1

        for i in range(num_patches_high):
            for j in range(num_patches_wide):
                y = i * (self.patch_length - self.overlap)
                x = j * (self.patch_length - self.overlap)

                if i == 0 and j == 0 or mode == "best":
                    patch = self.best_corr_patch(y, x)
                elif mode == "overlap":
                    patch = self.best_corr_overlap_patch(res, y, x)
                elif mode == "cut":
                    patch = self.best_corr_overlap_patch(res, y, x, alpha, level)
                    patch = self.min_cut_patch(patch, res, y, x)

                res[y:y + self.patch_length, x:x + self.patch_length] = patch

        return res

    def transfer_iter(self, n, mode="cut"):
        res = self.transfer(mode=mode)
        for i in tqdm(range(1, n), desc="Texture Transfer Progress"):
            alpha = 0.1 + 0.8 * i / (n - 1)
            self.patch_length = self.patch_length * 2 ** i // 3 ** i
            res = self.transfer(mode=mode, alpha=alpha, level=i, prior=res)

        return res
