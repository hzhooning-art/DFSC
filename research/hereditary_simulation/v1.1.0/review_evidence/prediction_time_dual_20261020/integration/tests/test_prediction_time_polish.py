import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from prediction_time_polish import polish_unresolved
from prediction_time_search import Information


class PolishTests(unittest.TestCase):
    def test_keeps_decisive_rows_and_numerical_failures(self):
        info = Information(1.,.1,(0.,),(.8,),(.01,))
        fixed = {'wait':.2,'state':'sufficient','route':'time_cell','lp_calls':0,
                 'solver_calls':0,'cell_variables':0}
        pending = {'wait':.11,'state':'unresolved','route':'inner_outer','lp_calls':4,
                   'solver_calls':4,'cell_variables':128,'outer':[.5,.9],'inner':[.7,.71]}
        original = {'rows':[pending,fixed],'lp_calls':4,'solver_calls':4,'cell_variables':128}
        def good(w,c):
            return {'outer':[.69,.72],'inner':[.7,.71],'programs':2,'solver_calls':2,
                    'cells':129,'fast_rate':2000.,'certificates':{'lower':{'cells':8},'upper':{'cells':7}}}
        result = polish_unresolved(info,1.,original,.025,.05,0.,[128],good,{'a':1})
        self.assertEqual(result['rows'][1],fixed)
        self.assertEqual(result['rows'][0]['state'],'sufficient')
        self.assertEqual(result['lp_calls'],6)
        self.assertEqual(original['rows'][0]['state'],'unresolved')
        def fail(*args):
            raise RuntimeError('intentional certificate failure')
        failed = polish_unresolved(info,1.,original,.025,.05,0.,[128],fail,{'a':1})
        self.assertEqual(failed['rows'][0]['state'],'unresolved')
        self.assertEqual(failed['rows'][1],fixed)
        untouched = polish_unresolved(info,0.,original,.025,.05,0.,[128],fail,{'a':1})
        self.assertEqual(untouched['rows'],original['rows'])
