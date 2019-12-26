import json
import sys

import sqlalchemy
from sqlalchemy import create_engine

from sqlalchemy.orm import sessionmaker

from sqlalchemy.sql import select
from db.models import Base, FrameGraphM, GameM, NodeM, PatchM, PatchGraphM
from db.data_types import BoundingBox, Shape

class DataStore:
    def __init__(self, file_path=None, echo=True):
        if file_path is None:
            self.engine = create_engine('sqlite:///:memory:', echo=echo)
        else:
            self.engine = create_engine(f'sqlite:///{file_path}', echo=echo)

        self.SessionFactory = sessionmaker(bind=self.engine)
        self._session = self.Session()

    def initialize(self, games_path):
        Base.metadata.create_all(self.engine)

        with open(games_path, 'r') as fp:
            games = json.load(fp)

        game_list = []
        game_list = sorted([*games['NES'], *games['SNES']])
        for i, g in enumerate(game_list):
            self._session.add(GameM(id=i, name=g, platform=g.split('-')[-1].upper()))
        self._session.commit()

    def query(self, *args, **kwargs):
        return self._session.query(*args, **kwargs)

    def Session(self):
        return self.SessionFactory()
