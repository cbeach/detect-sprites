from glob import glob
import os

import cv2
import numpy as np

from ..patch import Patch
from ..patch_graph import FrameGraph
from ..db.data_store import DataStore
from ..find import find_and_cull
from ..sprite_util import ensure_dir

TEST_PATH = os.path.dirname(os.path.realpath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../'))

ds = DataStore(f'{PROJECT_ROOT}/sprites/db/sqlite.db', games_path=f'{PROJECT_ROOT}/sprites/games.json', echo=False)
Patch.init_patch_db(ds=ds)

def get_test_cases(play_number, game):
    test_image_path = f'{TEST_PATH}/test_cases/{game}/{play_number}/test_images/'
    expected_image_path = f'{TEST_PATH}/test_cases/{game}/{play_number}/expected/'
    paths = [p.split('/')[-1] for p in glob(f'{TEST_PATH}/test_cases/{game}/{play_number}/test_images/*.png')]
    names = [p.split('.')[0] for p in paths]
    test_paths = [(test_image_path + i, expected_image_path + i) for i in paths]
    sprite_sets = {
        True: [[FrameGraph(cv2.imread(p, cv2.IMREAD_UNCHANGED), indirect=True, ds=ds) for p in
                glob(f'{TEST_PATH}/test_cases/{game}/sprites/test_sets/{name}/*')] for name in names],
        False: [[FrameGraph(cv2.imread(p, cv2.IMREAD_UNCHANGED), indirect=False, ds=ds) for p in
                glob(f'{TEST_PATH}/test_cases/{game}/sprites/test_sets/{name}/*')] for name in names]
    }

    # return test_case_names, test_cases, expected_values, sprite_sets
    return names, [cv2.imread(p, cv2.IMREAD_UNCHANGED) for p, _ in test_paths], [cv2.imread(p, cv2.IMREAD_UNCHANGED) for _, p in test_paths], sprite_sets

class TestCull:
    def setup_class(self):
        ensure_dir(f'{TEST_PATH}/failed-tests/')
        self.test_case_names, self.test_cases, self.expected, self.sprite_sets = get_test_cases(1000, 'SuperMarioBros-Nes')

    def test_images(self):
        ifgs = [FrameGraph(img.copy(), bg_color=img[0][0], indirect=True, ds=ds) for img in self.test_cases]
        dfgs = [FrameGraph(img.copy(), bg_color=img[0][0], indirect=False, ds=ds) for img in self.test_cases]

        for i, ifg in enumerate(ifgs):
            tc = self.test_cases[i]
            expected = self.expected[i]
            sprite_set = self.sprite_sets[True][i]
            culled_img = find_and_cull(ifg, sprite_set)
            try:
                assert(np.array_equiv(culled_img, expected))
            except AssertionError as err:
                cv2.imwrite(f'{TEST_PATH}/failed-tests/{self.test_case_names[i]}-original.png', tc)
                cv2.imwrite(f'{TEST_PATH}/failed-tests/{self.test_case_names[i]}-culled.png', culled_img)
                cv2.imwrite(f'{TEST_PATH}/failed-tests/{self.test_case_names[i]}-expected.png', expected)
                raise err



