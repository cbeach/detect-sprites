from collections import defaultdict
from enum import Enum

from pygame.locals import *
import pygame.locals as pglocals
from recordclass import recordclass

class Button(Enum):
    DESC = 1  # Descending
    DOWN = 2
    ASCE = 3  # Ascending
    UP = 4

ButtonState = recordclass('ButtonState', ['state', 'ticks'])

InputState = recordclass('InputState', ['keyboard', 'mouse'])
input_state = InputState(mouse=defaultdict(lambda: ButtonState(Button.UP), -1), keyboard=defaultdict(lambda: ButtonState(Button.UP), -1))
key_map = {getattr(pglocals, i): i for i in dir(pglocals) if i.startswith('K_')}
mod_map = {getattr(pglocals, i): i for i in dir(pglocals) if i.startswith('KMOD_')}
mouse_button_map = ['LEFT', 'MIDDLE', 'RIGHT', 'UP', 'DOWN', ]

def event_router(event, global_state, automatons):
    in_state = global_state.input_state
    keyboard = in_state.keyboard
    mouse = in_state.mouse

    # Pre process the keyboard state
    for button in keyboard.values() + mouse.values():
        if button.state == Button.DESC and button.ticks == 0:
            button.state = Button.DOWN
        elif button.state == Button.ASCE and button.ticks == 0:
            button.state = Button.UP
            button.ticks = -1
        if button.ticks > -1:
            button.ticks += 1

    # Process the event
    if event.type == KEYDOWN:
        key_name = key_map[event.key]
        keyboard[key_name].state = Button.DESC
        keyboard[key_name].ticks = 0

    elif event.type == KEYUP:
        key_name = key_map[event.key]
        keyboard[key_name].state = Button.ASCE
        keyboard[key_name].ticks = 0
    elif event.type == MOUSEBUTTONDOWN:
        button_name = key_map[event.button]
        mouse[button_name].state = Button.DESC
        mouse[button_name].ticks = 0
    elif event.type == MOUSEBUTTONUP:
        button_name = key_map[event.button]
        mouse[button_name].state = Button.ASCE
        mouse[button_name].ticks = 0

    keyboard_actions = {
        'K_w': automatons.nav.go_north,
        'K_s': automatons.nav.go_south,
        'K_d': automatons.nav.go_east,
        'K_a': automatons.nav.go_west,
    }
    mouse_actions = {
        'BUTTON_LEFT': automatons.nav.blink,
    }

    for k, v in keyboard.items():
        func = keyboard.get(keyboard_actions[k], lambda *args: None)
        func(event=event, nav_state=global_state)

    for k, v in mouse.items():
        func = mouse.get(mouse_actions[k], lambda *args: None)
        func(event=event, nav_state=global_state)






# elif event.type == KEYDOWN and (event.key == K_LEFT or event.key == K_h):
#    frame_number = frame_number - 1 if frame_number > 0 else pt.shape[0] - 1
#    update = True
# elif event.type == KEYDOWN and (event.key == K_RIGHT or event.key == K_l):
#    frame_number = frame_number + 1 if frame_number < pt.shape[0] - 1 else 0
#    update = True
# elif event.type == KEYDOWN and (event.key == K_UP or event.key == K_k):
#    frame_number = 0
#    play_number += 1
#    pt = get_playthrough(play_number, game=game)['raw']
#    update = True
# elif event.type == KEYDOWN and (event.key == K_DOWN or event.key == K_j):
#    frame_number = 0
#    play_number -= 1
#    pt = get_playthrough(play_number, game=game)['raw']
#    parse_frame(pt[frame_number])
#    update = True
# elif event.type == MOUSEBUTTONDOWN:
#    print('MOUSEBUTTONDOWN')
#    print('----------')
#    for i in dir(event):
#        if not i.startswith('__'):
#            print(f'attr: {i}')
#            print(f'  type : {type(getattr(event, i))}')
#            print(f'  value: {getattr(event, i)}')
#    print()
# elif event.type == KEYDOWN and event.key == K_ESCAPE:
#    selection = []
# elif event.type == MOUSEMOTION:
#    x, y = event.pos
#    frx, fry = get_frame_coords(x, y, scale)
# elif event.type == KEYDOWN:
#    print(key_map[event.key])
# elif event.type == MOUSEBUTTONDOWN:
#    print('MOUSEBUTTONDOWN')
#    print('----------')
#    for i in dir(event):
#        if not i.startswith('__'):
#            print(f'attr: {i}')
#            print(f'  type : {type(getattr(event, i))}')
#            print(f'  value: {getattr(event, i)}')
#    print()
# elif event.type == MOUSEBUTTONUP:
#    print('MOUSEBUTTONUP')
#    print('----------')
#    for i in dir(event):
#        if not i.startswith('__'):
#            print(f'attr: {i}')
#            print(f'  type : {type(getattr(event, i))}')
#            print(f'  value: {getattr(event, i)}')
#    print()
# elif event.type == MOUSEWHEEL:
#    print('MOUSEWHEEL')
#    print('----------')
#    for i in dir(event):
#        if not i.startswith('__'):
#            print(f'attr: {i}')
#            print(f'  type : {type(getattr(event, i))}')
#            print(f'  value: {getattr(event, i)}')
#    print()
