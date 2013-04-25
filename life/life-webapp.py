#!/usr/bin/env python
import time, json, Image
import flask
from flask import Flask, Response, redirect, request, url_for
import sys, os, os.path, string, random, math, time, urlparse
import itertools, collections
import life

app = Flask(__name__)
           
def simple_gen():
    "A simple generator for our streaming test"
    for i, c in enumerate("hello"*10):
        time.sleep(.1)  # an artificial delay
        yield i, c

@app.route('/test')
def test():
    """This is a simple test to make sure streaming is working. You should
    see incrementing numbers and cycling through the letters 'hello'. If
    you don't, you may need to enable threading in your wsgi server."""
    return Response(stream_template('test.html', data=simple_gen()))

def parse_parameters(request, param_info):
    param_dict = {}
    for param, desc in param_info.items():
        try:
            param_value = request.args[param]
        except (ValueError, IndexError):
            param_value = desc['default']
        
        param_dict[param] = desc['validator'](param_value)
    return param_dict

def random_chars(n):
    return ''.join(random.sample(
            string.ascii_uppercase+
            string.ascii_lowercase+
            string.digits, n))

def life_sim(params):
    sim = life.Life(params['grid_size'], params['p'])
    ## uimg is for user images
    uimg_path = os.path.join('uimg', random_chars(12))
    os.mkdir(os.path.join(app.static_folder, uimg_path))
    
    ## We need this so url_for() will work
    app.test_request_context('/life').push()
    
    def snapshot(dim):
        img_scale = float(dim)/params['grid_size']
        im = sim.draw(scale=img_scale)
        fname = random_chars(12) + '.png'
        outfile = open(os.path.join(app.static_folder, uimg_path, fname), 'w')
        im.save(outfile)
        outfile.close()
        return url_for('static', filename=os.path.join(uimg_path, fname))
    
    current_step = 0
    while current_step < params['num_steps']:
        ## This is just to make the incremental nature obvious
        time.sleep(0.5)
        
        steps_to_run = min(params['snapshot'], 
                           params['num_steps'] - current_step)
        #print "Running %i steps" % steps_to_run
        sim.run(steps_to_run)
        current_step += steps_to_run
        pct_done = float(current_step)/params['num_steps'] * 100
        datum = {'url' : snapshot(400), 
                'label': "Step: %i" % current_step,
                'pct'  : str(pct_done) + '%'}
        #print "data: %s\n" % json.dumps(datum)
        yield "data: %s\n\n" % json.dumps(datum)
        
    #print "data: byebye\n"
    yield "data: byebye\n\n"
        
@app.route('/life')
def param_test():
    param_info = collections.OrderedDict()
    param_info['grid_size'] = {
            'label': "Grid size",
            'default': 100,
            'validator': lambda x: int(max(1, x))}
    param_info['p'] = {   
            'label': "Initial random coverage",
            'default': 0.1,
            'validator': float}
    param_info['num_steps'] = {
            'label': "Steps to simulate",
            'default': 100,
            'validator': lambda x: int(max(1, x))}
    param_info['snapshot'] = {
            'label': "Steps between snapshots",
            'default': 10,
            'validator': lambda x: int(max(1, x))}
    
    if request.headers.get('accept') == 'text/event-stream':
        return Response(life_sim(parse_parameters(request, param_info)),
                        content_type='text/event-stream')
    else:
        return flask.render_template('life.html', params=param_info.items())  

if __name__ == "__main__":
    app.run(host='localhost', port=12345, debug=True, threaded=True)
