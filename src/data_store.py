import gzip
import pickle
from os import path

class BaseDB:
    def __init__(self, dirname, db_type, backend):
        self.dirname = dirname
        self.filename = path.join(self.dirname, f'{db_type}.{backend}.gz')
        self.backend = backend

    def _write_db(self):
        if self.backend == 'pickle':
            with gzip.GzipFile(self.filename, 'wb') as fp:
                pickle.dump(self, fp)

    def _load_db(self):
        if self.backend == 'pickle':
            with gzip.GzipFile(self.filename, 'rb') as fp:
                db = pickle.load(fp)
            return db

class PatchDB(BaseDB):
    __DB = None
    def __init__(self, dirname, backend='pickle'):
        super(PatchDB, self).__init__(dirname, 'patch_db', backend)

        if PatchDB.__DB is None:
            if path.isfile(self.filename):
                db = self._load_db()
                PatchDB.__DB = db.db
                self.dirname = db.dirname
                self.backend = db.backend
                self.filename = db.filename
            else:
                PatchDB.__DB = {
                    'patches': [],
                    'hash_to_patch_index': {},
                }
        self.db = PatchDB.__DB

    def __getstate__(self):
        return {
            'backend': self.backend,
            'dirname': self.dirname,
            'filename': self.filename,
            'db': self.db,
        }

    def __setstate__(self, data):
        self.backend = data['backend']
        self.dirname = data['dirname']
        self.filename = data['filename']
        self.db = data['db']

    def add_patch(self, patch):
        patch_hash = hash(patch)
        if patch_hash not in self.db['hash_to_patch_index']:
            self.db['patches'].append(patch)
            self.db['hash_to_patch_index'][patch_hash] = patch
        return self.db['hash_to_patch_index'][patch_hash]

    def get_patch(self, index):
        return self.db['patches'][index]

    def find_patch(self, patch_hash):
        if patch_hash in self.db['hash_to_patch']:
            return self.db['patches'][index]
        else:
            raise KeyError(f'patch does not exist. got {patch_hash}')

    def write(self):
        self._write_db()


class GraphDB(BaseDB):
    def __init__(self, dirname, backend='pickle'):
        super(PatchDB, self).__init__(dirname, 'patch_db', backend, shape={
            'play_number': {},
        })

    def add_frame(self, patch):
        patch_hash = hash(patch)
        if patch_hash not in self.db['hash_to_patch_index']:
            self.db['patches'].append(patch)
            self.db['hash_to_patch_index'][patch_hash] = patch
        self._write_db()
        return self.db['hash_to_patch_index'][patch_hash]
