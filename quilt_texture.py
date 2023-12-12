import numpy as np
import math
from skimage import io, util
import heapq


class QuiltTexture:
    def __init__(self, texture, patch_length):
        self.texture = util.img_as_float(texture)
        self.patch_length = patch_length
        self.overlap = patch_length // 6
        self.num_patches_high = None
        self.num_patches_wide = None
        self.result = None

    def random_patch(self):
        h, w, _ = self.texture.shape
        i = np.random.randint(h - self.patch_length)
        j = np.random.randint(w - self.patch_length)
        return self.texture[i:i + self.patch_length, j:j + self.patch_length]

    def l2_overlap_diff(self, patch, res, y, x):
        error = 0

        if x > 0:
            left = patch[:, :self.overlap] - res[y:y + self.patch_length, x:x + self.overlap]
            error += np.sum(left ** 2)

        if y > 0:
            up = patch[:self.overlap, :] - res[y:y + self.overlap, x:x + self.patch_length]
            error += np.sum(up ** 2)

        if x > 0 and y > 0:
            corner = patch[:self.overlap, :self.overlap] - res[y:y + self.overlap, x:x + self.overlap]
            error -= np.sum(corner ** 2)

        return error

    def random_best_patch(self, res, y, x):
        h, w, _ = self.texture.shape
        errors = np.zeros((h - self.patch_length, w - self.patch_length))

        for i in range(h - self.patch_length):
            for j in range(w - self.patch_length):
                patch = self.texture[i:i + self.patch_length, j:j + self.patch_length]
                e = self.l2_overlap_diff(patch, res, y, x)
                errors[i, j] = e

        i, j = np.unravel_index(np.argmin(errors), errors.shape)
        return self.texture[i:i + self.patch_length, j:j + self.patch_length]

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

    def quilt(self, num_patches, mode="cut", sequence=False):
        self.num_patches_high, self.num_patches_wide = num_patches
        h = (self.num_patches_high * self.patch_length) - (self.num_patches_high - 1) * self.overlap
        w = (self.num_patches_wide * self.patch_length) - (self.num_patches_wide - 1) * self.overlap
        self.result = np.zeros((h, w, self.texture.shape[2]))

        for i in range(self.num_patches_high):
            for j in range(self.num_patches_wide):
                y = i * (self.patch_length - self.overlap)
                x = j * (self.patch_length - self.overlap)

                if i == 0 and j == 0 or mode == "random":
                    patch = self.random_patch()
                elif mode == "best":
                    patch = self.random_best_patch(self.result, y, x)
                elif mode == "cut":
                    patch = self.random_best_patch(self.result, y, x)
                    patch = self.min_cut_patch(patch, self.result, y, x)

                self.result[y:y + self.patch_length, x:x + self.patch_length] = patch

                if sequence:
                    io.imshow(self.result)
                    io.show()

        return self.result

    def quilt_size(self, shape, mode="cut", sequence=False):
        h, w = shape
        num_patches_high = math.ceil((h - self.patch_length) / (self.patch_length - self.overlap)) + 1 or 1
        num_patches_wide = math.ceil((w - self.patch_length) / (self.patch_length - self.overlap)) + 1 or 1
        result = self.quilt((num_patches_high, num_patches_wide), mode, sequence)
        return result[:h, :w]
