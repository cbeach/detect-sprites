import json

from sqlalchemy import create_engine
from sqlalchemy.orm import relationship
from sqlalchemy.schema import ForeignKeyConstraint
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Table, Column, MetaData, ForeignKey
from sqlalchemy import Boolean, Integer, LargeBinary, String

from .data_types import BoundingBox, Color, Coord, FrameID, Mask, Shape, encode_frame_id, decode_frame_id

Base = declarative_base()

class PatchM(Base):
    __tablename__ = 'patches'

    id = Column(Integer, primary_key=True)
    mask = Column(Mask)
    shape = Column(Shape)
    indirect = Column(Boolean)

    def __repr__(self):
        return f"<Patch(id={self.id}, shape={[int(i) for i in self.shape]}, indirect={self.indirect}, mask=\n{self.mask})>"

class NodeM(Base):
    __tablename__ = 'nodes'

    id = Column(Integer, primary_key=True)
    patch = Column(None, ForeignKey('patches.id'))
    color = Column(Color)
    bb = Column(BoundingBox)
    game_id = Column(Integer)
    play_number = Column(Integer)
    frame_number = Column(Integer)
    edge_of_frame = Column(Boolean)
    bg_edge = Column(Boolean)

    def __repr__(self):
       return f"<Node(id='{self.id}', patch='{self.patch}', color='{self.color}', bounding_box={self.bb}, graph_id={[self.game_id, self.play_number, self.frame_number]})>"

class EdgeM(Base):
    __tablename__ = 'edges'
    global_id = Column(Integer, autoincrement=True, primary_key=True)
    left_id = Column(Integer, ForeignKey('nodes.id'), index=True)
    right_id = Column(Integer, ForeignKey('nodes.id'), index=True)
    x_offset = Column(Integer)
    y_offset = Column(Integer)

    def __repr__(self):
       return f"<Edge(id='({self.left_id}, {self.right_id}', offset='({self.x_offset}, {self.y_offset})')>"

class GameM(Base):
    __tablename__ = 'games'

    id = Column(Integer, autoincrement=True, primary_key=True)
    name = Column(String)
    platform = Column(String)

    def __repr__(self):
        return f'<Game(id={self.id}, name={self.name})>'

#class PatchGraphM(Base):
#    __tablename__ = 'patch_graphs'
#    id = Column(Integer, primary_key=True)
#    #tlc = Column(Coord)
#    nodes = relationship('NodeM')
#    frame_graph = Column(None, ForeignKey('frame_graphs.id'))

#class FrameGraphM(Base):
#    __tablename__ = 'frame_graphs'
#
#    game = Column(None, ForeignKey('games.id'))#, primary_key=True)
#    play_number = Column(Integer, primary_key=True)
#    frame_number = Column(Integer, primary_key=True)
#    nodes = relationship('NodeM', primaryjoin="and_(NodeM.game_id==FrameGraphM.game, and_(NodeM.play_number==FrameGraphM.play_number, NodeM.frame_number==FrameGraphM.frame_number))")
#
#    def __repr__(self):
#       return f"<FrameGraph(game='{self.game}', play_number='{self.play_number}', frame_number='{self.frame_number}', nodes={self.nodes})>"

