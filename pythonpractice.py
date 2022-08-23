#!/usr/bin/env python3
'''
DATA 419 -- Assignment #0.5 support file
Stephen Davies, University of Mary Washington, fall 2022
'''

import sys
import importlib

''' 
To run this program, make sure you have a Python file in the current directory
whose name is exactly your lower-case UMW Net ID (e.g., "jsmith19") followed
by the string "_pythonpractice.py".

At any point (when you're done or earlier), you can run it via:

$ python3 pythonpractice.py jsmith19
'''

def print_usage():
    print('Usage: pythonpractice.py UMWNetID.')

if len(sys.argv) != 2:
    print_usage()
    sys.exit(1)

try:
    exec(open("./" + sys.argv[1] + "_pythonpractice.py").read())
except Exception as err:
    print(str(err))
    sys.exit(2)

print("Testing {}...".format(sys.argv[1] + '_pythonpractice.py'))

points = 0

the_vars = {
    'WALL': [['E']],
    'R2': list('D2'),
    'HAL': 9000,
    'K': {'2SO'},
    'C': [{'3PO'}],
    'BB': '8',
    'L': [ 3 ],
    'Poppins': set(list('supercalifragilisticexpialidocious')),
    'Galactica': { 'Karl':'Helo', 'Kara':'Starbuck', 'Lee':'Apollo', 
        'Sharon':'Boomer', 'Marge':'Racetrack' }
}

for var,val in the_vars.items():
    all_correct = True
    if (var not in globals() or
        type(val) != type(globals()[var]) or
        val != globals()[var]):
            print(f"Variable {var} incomplete or incorrect.")
            all_correct = False
            break
if all_correct:
    points += 2
    print("Variables (part 1) correct! +2")

if all_correct:
    names = ['Germanna_levels','UMW_levels','MaryWash_levels','CNU_levels']
    all_present = True
    for name in names:
        if name not in globals():
            print(f"Variable {name} incomplete or incorrect.")
            all_present = False
            break
        globals()[name] = globals()[name]
    if all_present:
        if (Germanna_levels != UMW_levels and UMW_levels is MaryWash_levels and
            UMW_levels == CNU_levels and UMW_levels is not CNU_levels):
            points += 2
            print("Variables (part 2) correct! +2")
        else:
            print("Variables (part 2) incomplete/incorrect.")
    

if 'plus2' in globals():
    if plus2(27) == 29:
        points += 1
        print("plus2() correct! +1")
    else:    
        print("plus2() incorrect.")
else:
    print("(plus2() incomplete.)")




if 'gimme_dat_set' in globals():
    thing1 = gimme_dat_set()
    thing2 = gimme_dat_set()
    if thing1 is thing2 and thing1 == {'Pris','Leon','Roy'}:
        points += 2
        print("gimme_dat_set() correct! +1")
    else:    
        print("gimme_dat_set() incorrect.")
else:
    print("(gimme_dat_set() incomplete.)")


if 'gimme_set_like_dat' in globals():
    thing1 = gimme_set_like_dat()
    thing2 = gimme_set_like_dat()
    if (thing1 is not thing2 and thing1 == thing2 and
        thing1 == {'Neo','Morpheus','Trinity'}):
        points += 2
        print("gimme_set_like_dat() correct! +1")
    else:    
        print("gimme_set_like_dat() incorrect.")
else:
    print("(gimme_set_like_dat() incomplete.)")


if 'center' in globals():
    if (center('spiderman') == 'e' and
       center('batman') == 'tm' and
       center('') == None and
       center('thor') == 'ho'):
        points += 2
        print("center() correct! +2")
    else:    
        print("center() incorrect.")
else:
    print("(center() incomplete.)")


if 'middlest' in globals():
    middlest_func = globals()['middlest']
    if (middlest_func(5,9,1) == 5 and
       middlest_func(1,22,3) == 3 and
       middlest_func(1,2,3) == 2 and
       middlest_func(2,2,2) == 2):
        points += 2
        print("middlest() correct! +1")
    else:    
        print("middlest() incorrect.")
    
else:
    print("(middlest() incomplete.)")


if 'nuke_last' in globals():
    stuff = list('abcdefedbcaerugioaidsfosa')
    nuke_last(stuff,'o')
    nuke_last(stuff,'a')
    nuke_last(stuff,'s')
    nuke_last(stuff,'e')
    nuke_last(stuff,'r')
    nuke_last(stuff,'i')
    if ''.join(stuff) == 'abcdefedbcaugioadsf':
        points += 2
        print("nuke_last() correct! +2")
    else:    
        print("stuff = {}".format(''.join(stuff)))
        print("nuke_last() incorrect.")
else:
    print("(nuke_last() incomplete.)")


if 'tack_on_end' in globals():
    tack_on_end_func = globals()['tack_on_end']
    some_thing = ['a','b']
    tack_on_end_func(some_thing,'c')
    tack_on_end_func(some_thing,'d',2)
    tack_on_end_func(some_thing,['e','f'],2)
    tack_on_end_func(some_thing,['g','h','i'])
    if some_thing == ['a','b','c','d','d','e','f','e','f','g','h','i']:
        points += 2
        print("tack_on_end() correct! +1")
    else:    
        print("tack_on_end() incorrect.")
    
else:
    print("(tack_on_end() incomplete.)")


if 'wondrous_count' in globals():
    wondrous_count_func = globals()['wondrous_count']
    test_vals = { 1:0, 6400:31, 6401:124, 99999:226, 6171:261, 6170:36,
        75128138246:225, 75128138247:1228 }
    if all([wondrous_count_func(tv) == tw for tv,tw in test_vals.items()]):
        points += 2
        print("wondrous_count() correct! +2")
    else:    
        print("wondrous_count() incorrect.")
    
else:
    print("(wondrous_count() incomplete.)")


if 'unique_vals' in globals():
    uv_func = globals()['unique_vals']
    uv = uv_func({ 'Malone':32, 'Ruth':3, 'Favre':4, 'Jordan':23, 
        'Sandberg':23, 'Kobe':24, 'Jeter':2, 'Brown':32, 'Magic':32, 
        'Elway':7, 'Koufax':32, 'Mantle':7 })
    if (type(uv) == list and len(uv) == 7 and
        all([ x in uv for x in [2,3,4,7,23,24,32]])):
        points += 1
        print("unique_vals() correct! +1")
    else:    
        print("unique_vals() incorrect.")
else:
    print("(unique_vals() incomplete.)")

print("You've earned {}/20 XP!".format(points))
