# EOD

    Saving sprites as images now. Need to save them in a more sensible format.
    Need to create the "UI" for

# Thoughts

Start graphlet mining at nodes that aren't at a border (background or screen edge)

# Serialize and store
## Use sqlalchemy to store frame graphs

  [x] Construct Schema - Done
  [x] Store \_patch objects in the db
    [x] [De]Serialize the mask data
        * Code is in db.data_types.Mask
  [.] Design a data model that can be quickly reconstructed 
      - Should be based on sub-graphs
  [.] Switch to postgresql

## Data Model
### Patch
  - Mask
  - Shape

### Node
  - Patch (foreignkey(Patch))
  - Color
  - Top left corner

### Edge
  - Node (left, foreignkey(Node))
  - Node (right, foreignkey(Node))
  - x & y offset
    * (x\_left - x\_right, y\_left - y\_right)

### PatchGraph
  - RootNode (foreignkey(Node))
  - IsSprite (boolean)
  - Palette (Do I want?)


### Frame
  - Game (foreighkey(game))
  - Frame number (int)
  - Play number (int)
  - List of PatchGraphs (many-to-many or one-to-many?)

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


