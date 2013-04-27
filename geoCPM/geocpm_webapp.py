#!/usr/bin/env python
import time, json, Image
import flask
from flask import Flask, Response, redirect, request, url_for
import sys, os, os.path, string, random, math, time, urlparse
import itertools, collections, string
import geocpm_core

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

def geocpm_sim(params):
    """
    Simulation generator expecting the following arguments:
    ['num_steps', 'T_initial', 'T_final', 'K', 'snapshot', 
     'lambda_comp', 'lambda_pop', 'dataset']
    """
    data_path = os.path.join(app.static_folder, 'data', params['dataset'])
    data_file = [os.path.splitext(fname)[0] for fname in os.listdir(data_path) 
                    if os.path.splitext(fname)[1] == '.shp'].pop()
    data_frame = geocpm_core.load_shapefile(os.path.join(data_path, data_file))
    neighbor_graph = geocpm_core.make_graph(data_frame)
    print 'K:', params['K']
    sim = geocpm_core.Simulation(neighbor_graph, params['K'], data_frame,
                        lambdas=(params['lambda_comp'], params['lambda_pop']))
    
    uimg_path = os.path.join('uimg', random_chars(12))
    os.mkdir(os.path.join(app.static_folder, uimg_path))
    app.test_request_context('/geocpm').push()
                    
    def snapshot(dim):
        im = sim.get_image(x_dim=dim)
        # boilerplate follows
        fname = random_chars(12) + '.png'
        outfile = open(os.path.join(app.static_folder, uimg_path, fname), 'w')
        im.save(outfile)
        outfile.close()
        return url_for('static', filename=os.path.join(uimg_path, fname))
        
    datum = {'url' : snapshot(400), 
            'label': "Initial Condition",
            'pct'  : "0%"}
    print "data: %s\n" % json.dumps(datum)
    yield "data: %s\n\n" % json.dumps(datum)
    
    T_i, T_f, num_steps = params['T_initial'], params['T_final'], params['num_steps']
    decay_constant = math.log(T_i/float(T_f))/num_steps
    expo_decay = lambda t: T_i * math.exp(-decay_constant * t)
    lin_decay = lambda t: - t*(T_i - T_f)/float(num_steps) + T_i
    annealing_schedule = map(lin_decay, range(0, num_steps, params['snapshot']))
    print "Annealing schedule:", annealing_schedule 
        
    for T in annealing_schedule:
        steps_to_run = min(params['snapshot'], 
                           params['num_steps'] - sim.mcs)
        
        sim.T = T
        print "T:", sim.T
        sim.run(steps_to_run, debug=1)
        pct_done = float(sim.mcs)/params['num_steps'] * 100
        datum = {'url' : snapshot(400), 
                'label': "MCS: %i, T: %i" % (sim.mcs, round(sim.T)),
                'pct'  : str(pct_done) + '%'}
        print "data: %s\n" % json.dumps(datum)
        yield "data: %s\n\n" % json.dumps(datum)
        
    print "data: byebye\n"
    yield "data: byebye\n\n"
    
@app.route('/geocpm')
def param_test():
    param_info = collections.OrderedDict()
    param_info['num_steps'] = {   
            'label': "Number of steps",
            'default': 100,
            'validator': lambda x: max(1, int(x))}
    param_info['snapshot'] = {
            'label': "Steps between snapshots",
            'default': 10,
            'validator': lambda x: max(1, int(x))}
    param_info['T_initial'] = {
            'label': "Initial temperature T",
            'default': 200,
            'validator': lambda x: max(1.0, float(x))}
    param_info['T_final'] = {   
            'label': "Final temperature T",
            'default': 50,
            'validator': lambda x: max(1.0, float(x))}
    param_info['lambda_comp'] = {
            'label': "Prefactor for compactness objective",
            'default': 5.0,
            'validator': float}
    param_info['lambda_pop'] = {
            'label': "Prefactor for population objective",
            'default': 1.0,
            'validator': float}
    param_info['dataset'] = {
            'label': "Data set",
            'choices': ["Indiana"],
            'validator': string.lower}
    param_info['K'] = {
            'label': "Number of districts",
            'default': 9,
            'validator': lambda x: int(max(1, x))}
    
    if request.headers.get('accept') == 'text/event-stream':
        return Response(geocpm_sim(parse_parameters(request, param_info)),
                        content_type='text/event-stream')
    else:
        return flask.render_template('geocpm.html', 
            app_name='geoCPM',
            params=param_info.items())  

if __name__ == "__main__":
    app.run(host='localhost', port=12345, debug=True, threaded=True)
