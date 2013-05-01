import collections
import numpy as np
import time

class FComparator(object):
    def __init__(self):
        self.num_runs = 0
        self.function_dict = collections.OrderedDict({})
    
    def add(self, f, args):
        assert hasattr(f, '__call__')
        assert args.__iter__
        self.function_dict[f.__name__] = {'funct': f, 'args':args, 'times':[], 'results':[]}
        
    def remove(self, f):
        if hasattr(f, '__call__'):
            f_key = f.__name__
        else:
            f_key = f
            
        del self.function_dict[f_key]
        
    def clear_runs(self):
        for k in self.function_dict:
            self.function_dict[k]['times'] = []
            self.function_dict[k]['results'] = []
            self.num_runs = 0
        
    def run(self, num_runs=1, sort_key=None):
        if sort_key == None:
            keys = self.function_dict.keys()
        else:
            keys = sorted(self.function_dict.keys(), key=sort_key)
        
        for _ in xrange(num_runs):
            self.num_runs += 1
            for k in keys:
                dict_entry = self.function_dict[k]
                (funct, args) = (dict_entry['funct'], 
                                 dict_entry['args'])
                tic = time.time()
                dict_entry['results'].append(funct(*args))
                dict_entry['times'].append(time.time() - tic)
        
        print "Data after %i runs:" % self.num_runs
        format_str = "%-30.30s  %-12.12s  "
        print (format_str + "(%+7s/%+7s/%+7s)") % ('Name', 'Result', 'min', 'mean', 'max')
        print "-"*72
        format_str += "(%0.5f/%0.5f/%0.5f)"
        for k in keys:
            de = self.function_dict[k]
            print format_str % (k, de['results'][-1], min(de['times']), np.mean(de['times']), max(de['times']))
            
    def compare(self, f1, f2):
        f_name = []
        for f in (f1, f2):
            if hasattr(f, '__call__'):
                f_name.append(f.__name__)
            else:
                f_name.append(f)
        times_faster = np.mean(self.function_dict[f_name[1]]['times'])/np.mean(self.function_dict[f_name[0]]['times'])
        if times_faster < 1:
            (f_name[0], f_name[1]) = (f_name[1], f_name[0])
            times_faster = 1./times_faster
        print "On average, %s is %i times faster than %s." % (f_name[0], times_faster, f_name[1])
