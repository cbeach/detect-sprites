import json
import multiprocessing as mp
from multiprocessing import Pool, Value, Array
import sys

from flask import Flask, request, send_from_directory, jsonify

from patch import Patch, frame_edge_nodes, frame_edge_node, background_node, background_nodes
from patch_graph import FrameGraph
from sprite_util import show_image, get_image_list, get_playthrough, load_indexed_playthrough, sort_colors, patch_encoder, node_encoder, graph_encoder
import db
from db.data_store import DataStore
from db.models import NodeM, PatchM

mp.set_start_method('fork')
# set the project root directory as the static folder, you can set others.
app = Flask(__name__, static_url_path='')
IMAGE_DIR='db/SuperMarioBros-Nes/images/1000/'

ds = DataStore('db/sqlite.db', games_path='./games.json', echo=False)
Patch.init_patch_db(ds)

# There are 1,670 frames in play number 1000.
# I'm only working on play number 1000 for now, hence the magic number
processed_frames = {
    True: [None] * 1670, # indirect
    False: [None] * 1670, # direct
}

@app.route('/js/<path:path>')
def send_js(path):
    return send_from_directory('static/js', path)

@app.route('/png/<path:path>')
def frame_image(path):
    return send_from_directory(IMAGE_DIR, path)

@app.route('/frame/<ntype>/<int:frame_number>', methods=['GET'])
def get_frame(frame_number, ntype='indirect'):
    ntype = ntype == 'indirect'
    frame = FrameGraph.from_raw_frame('SuperMarioBros-Nes', play_number=1000, frame_number=frame_number, indirect=ntype, ds=ds) if processed_frames[ntype][frame_number] is None else processed_frames[ntype][frame_number]
    graph = graph_encoder(frame)
    print(graph['shape'])
    return jsonify(graph)


if __name__ == '__main__':
    #ntype = True
    #frame_number = 113
    #frame = FrameGraph.from_raw_frame('SuperMarioBros-Nes', play_number=1000, frame_number=frame_number, indirect=ntype, ds=ds) if processed_frames[ntype][frame_number] is None else processed_frames[ntype][frame_number]

    #graph = graph_encoder(frame)
    #jg = json.dumps(graph)
    #print(jg)
    app.run(debug=True)
