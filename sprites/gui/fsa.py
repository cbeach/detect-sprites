from recordclass import recordclass
from statemachine import StateMachine, Transition, State

NavigationState = recordclass('NavigationState', ['state', 'cursor_pos'])

class NavigationAutomata(StateMachine):
    still = State('still', initial=True)
    n = State('north')
    s = State('south')
    e = State('east')
    w = State('west')
    ne = State('north_east')
    se = State('south_east')
    nw = State('north_west')
    sw = State('south_west')
    left_click = State('left_click')

    go_north = still.to(n)
    go_south = still.to(s)
    go_east = still.to(e)
    go_west = still.to(w)
    go_ne = still.to(ne)
    go_se = still.to(se)
    go_nw = still.to(nw)
    go_sw = still.to(sw)
    blink = still.to(left_click)
    halt = n.to(still) | s.to(still) | e.to(still) | w.to(still) | ne.to(still) | se.to(still) | nw.to(still) | sw.to(still) | left_click.to(still)

    def on_go_north(self, *args, **kwargs):
        print('on_go_north')
        kwargs['global_state'].nav_state.cursor_pos.x += 1
        return kwargs['global_state']

    def on_go_south(self, *args, **kwargs):
        print('on_go_south')
        kwargs['global_state'].nav_state.cursor_pos.x -= 1
        return kwargs['global_state']

    def on_go_east(self, *args, **kwargs):
        print('on_go_east')
        kwargs['global_state'].nav_state.cursor_pos.y -= 1
        return kwargs['global_state']

    def on_go_west(self, *args, **kwargs):
        print('on_go_west')
        kwargs['global_state'].nav_state.cursor_pos.y += 1
        return kwargs['global_state']

    def on_go_ne(self, *args, **kwargs):
        print('on_go_ne')
    def on_go_se(self, *args, **kwargs):
        print('on_go_se')
    def on_go_nw(self, *args, **kwargs):
        print('on_go_nw')
    def on_go_sw(self, *args, **kwargs):
        print('on_go_sw')

    def on_blink(self, *args, **kwargs):
        print('on_go_jump')
        kwargs['global_state'].nav_state.cursor_pos.x = kwargs['event'].pos[0]
        kwargs['global_state'].nav_state.cursor_pos.y = kwargs['event'].pos[1]

    def on_reset(self):
        print('on_go_reset')
        pass
