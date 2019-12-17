# EOD

    I'm working on serializing the sub-patches. The required code is being added to 
    Mask.process_bind_param and Mask.process_result_value in db/data_types.py.
    After that I need to work my way up the data structures, serializing as I go.

# Serialize and store
## Use sqlalchemy to store frame graphs

  [x] Construct Schema - Done
  [.] Store \_patch objects in the db
    [.] [De]Serialize the mask data
        * Code is in db.data_types.Mask


# Sprite isolation strategies

  * Mark nodes as "edge nodes" if they touch the background or edge of the screen
  * Create hash function for graph nodes
    * Combines patch hash with neighbor's hashes and relative offset
  * Small (< 300px) disjoint sub-graphs could be single sprites
  * Use both direct and indirect image traversal for patch parsing
  * Use frame palette as guiding heuristic
  * Find recurring sub-graphs 
    * Eliminate sub-graphs based on relative offset image.
      * Eg. Mario running across ground; Mario changes position from frame to frame and therefore
            is regarded as a separate subgraph

# Write a front end for basic manual sprite isolation

  * Very simple - canvas with a few buttons
  * Backend API
    * GET /sub-graph
    * POST /is-minimum
      - Body: {
        "subgraph_id": string,
        "is_min": boolean,
      }
    * PUT /sprite
      - Body: [ patches ]

# Done
## Refactor
### Patch Graph

  * A Patch Graph is a collection of graphs
    [.] Patch Graph copy constructor
  x Combine the PatchGraph and Frame classes (move Frame into PatchGraph)

### Node Class
  * Contains helper functions such as patch color and position
    * Move patch color and position out of patch and into Node

