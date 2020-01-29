import json
import sys

import sqlalchemy
from sqlalchemy import create_engine

from sqlalchemy.orm import sessionmaker

from sqlalchemy.sql import select
from .models import Base, GameM, NodeM, PatchM#, GraphletM, FrameGraphM
from .data_types import BoundingBox, Shape

class DataStore:
    def __init__(self, file_path=None, echo=True, games_path=None):
        if file_path is None:
            self.engine = create_engine('sqlite:///:memory:', echo=echo)
        else:
            self.engine = create_engine(f'sqlite:///{file_path}', echo=echo)

        self.SessionFactory = sessionmaker(bind=self.engine)
        self._session = self.Session()
        self.games_path = games_path
        #if len(sqlalchemy.inspect(self.engine).get_table_names()) == 0:
        #    self.initialize(games_path)

    def initialize(self):
        print('initializing database')
        Base.metadata.create_all(self.engine, checkfirst=True)

        with open(self.games_path, 'r') as fp:
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
